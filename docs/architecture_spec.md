# Financial Entity Sentiment Analysis: Production Architecture Specification

## Overview

This document specifies a production architecture for entity-level sentiment analysis on Bloomberg financial news articles. The system uses a shared Longformer encoder with modular task heads for NER, coreference resolution, and sentiment scoring.

**Key Design Decisions:**
- **Encoder**: Longformer (4,096 tokens) — chosen for runtime-configurable entity attention masking
- **Sentiment Labels**: Claude Haiku generates continuous [-1, 1] scores via knowledge distillation
- **Training**: Regression head with MSE loss on Haiku-generated labels
- **Target Latency**: Sub-5 seconds per article on Nvidia A6000

---

## Why Longformer Over ModernBERT

ModernBERT offers 2-4x speed improvement and 8,192 token context, but its **alternating attention pattern is fixed at architecture level**:

| Layer | Attention Type | Token Visibility |
|-------|----------------|------------------|
| 1, 2 | Local | 128 nearest tokens only |
| 3 | Global | All tokens |
| 4, 5 | Local | 128 nearest tokens only |
| 6 | Global | All tokens |

This pattern cannot be modified at runtime. For entity-level sentiment, we need specific entity tokens to attend globally to capture sentiment expressed anywhere in the article.

**Longformer** provides `global_attention_mask` parameter allowing runtime control:

```python
# Entity tokens attend to ALL tokens; ALL tokens attend to entity tokens
global_attention_mask = torch.zeros_like(input_ids)
global_attention_mask[:, 0] = 1  # CLS token
global_attention_mask[:, entity_start:entity_end] = 1  # Target entity
```

**Context Length Justification**: 95%+ of Bloomberg news articles fit within 4,096 tokens. Standard news: 400-1,200 tokens; long analysis: 2,000-3,000 tokens.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Article                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Longformer Encoder                            │
│                 (allenai/longformer-base-4096)                   │
│                                                                  │
│  • Global attention on CLS + entity positions                    │
│  • Local sliding window (512 tokens) elsewhere                   │
│  • Output: (batch, seq_len, 768) hidden states                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│     NER Head      │ │  Coref Head     │ │  Sentiment Head     │
│                   │ │                 │ │                     │
│ • Token classif.  │ │ • FastCoref     │ │ • Entity attention  │
│ • BIO tagging     │ │ • Span pairs    │ │ • Regression [-1,1] │
│ • 7 entity types  │ │ • Clustering    │ │ • MSE loss          │
└───────────────────┘ └─────────────────┘ └─────────────────────┘
                │               │               │
                └───────────────┼───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output per Entity                           │
│  {                                                               │
│    "entity": "Apple Inc.",                                       │
│    "type": "COMPANY",                                            │
│    "mentions": [(12, 22), (145, 156), (389, 400)],              │
│    "coreference": ["Apple Inc.", "the company", "AAPL"],        │
│    "sentiment_score": 0.72                                       │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Shared Encoder

**Model**: `allenai/longformer-base-4096`
- Parameters: 149M
- Hidden size: 768
- Attention heads: 12
- Layers: 12
- Context: 4,096 tokens
- Local attention window: 512 tokens

**Loading**:
```python
from transformers import LongformerModel, LongformerTokenizer

encoder = LongformerModel.from_pretrained("allenai/longformer-base-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
```

**Global Attention Configuration**:
```python
def create_global_attention_mask(input_ids, entity_positions):
    """
    Create attention mask where entity tokens attend globally.
    
    Args:
        input_ids: (batch, seq_len) token IDs
        entity_positions: List of (start, end) tuples per batch item
    
    Returns:
        global_attention_mask: (batch, seq_len) with 1s for global attention
    """
    batch_size, seq_len = input_ids.shape
    global_attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # CLS token always gets global attention
    global_attention_mask[:, 0] = 1
    
    # Entity positions get global attention
    for batch_idx, positions in enumerate(entity_positions):
        for start, end in positions:
            global_attention_mask[batch_idx, start:end] = 1
    
    return global_attention_mask
```

### 2. NER Head

**Purpose**: Identify financial entities when not pre-provided

**Architecture**: Token classification with BIO tagging

**Entity Types** (7 classes + O):
- `COMPANY`: Apple Inc., Goldman Sachs
- `TICKER`: AAPL, GS
- `PERSON`: Tim Cook, Jamie Dimon
- `ORG`: Federal Reserve, SEC
- `MONEY`: $1.5 billion, €500 million
- `PERCENT`: 15%, 2.5 percentage points
- `DATE`: Q3 2024, fiscal year 2025

**Implementation**:
```python
class NERHead(nn.Module):
    def __init__(self, hidden_size=768, num_labels=15):  # 7 types * 2 (B/I) + O
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, seq_len, num_labels)
        """
        x = self.dropout(encoder_output)
        logits = self.classifier(x)
        return logits
```

**Alternative - Zero-Shot with GLiNER**:
For flexibility without labeled NER data, use GLiNER:
```python
from gliner import GLiNER

ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
entities = ner_model.predict_entities(
    text,
    labels=["company", "ticker", "person", "organization", "money", "percent", "date"]
)
```

### 3. Coreference Head

**Purpose**: Link entity mentions (pronouns, abbreviations, aliases)

**Model**: FastCoref (`biu-nlp/f-coref`)
- Speed: 25 seconds for 2,800 documents on V100
- Accuracy: 78.5 F1 on OntoNotes

**Integration**:
```python
from fastcoref import FCoref

class CorefHead:
    def __init__(self, device='cuda:0'):
        self.model = FCoref(device=device)
    
    def predict(self, texts):
        """
        Args:
            texts: List of article strings
        Returns:
            List of cluster lists per article
        """
        predictions = self.model.predict(texts=texts)
        return [pred.get_clusters() for pred in predictions]
```

**Financial Domain Augmentation**:
FastCoref handles pronouns/noun phrases. Augment with ticker-company mapping:

```python
# Load from SEC EDGAR or maintain manually
TICKER_TO_COMPANY = {
    "AAPL": ["Apple Inc.", "Apple"],
    "GOOGL": ["Alphabet Inc.", "Google"],
    "MSFT": ["Microsoft Corporation", "Microsoft"],
    # ... extend as needed
}

def augment_coref_with_tickers(coref_clusters, text):
    """Add ticker-company links to coreference clusters."""
    augmented = []
    for cluster in coref_clusters:
        expanded_cluster = list(cluster)
        for mention in cluster:
            # Check if mention is a known ticker
            if mention.upper() in TICKER_TO_COMPANY:
                expanded_cluster.extend(TICKER_TO_COMPANY[mention.upper()])
            # Check if mention matches a company name
            for ticker, names in TICKER_TO_COMPANY.items():
                if mention in names:
                    expanded_cluster.append(ticker)
                    expanded_cluster.extend(names)
        augmented.append(list(set(expanded_cluster)))
    return augmented
```

### 4. Sentiment Head (Knowledge Distillation from Haiku)

**Training Approach**: 
1. Generate continuous [-1, 1] sentiment labels using Claude Haiku
2. Train regression head with MSE loss on Haiku labels

**Label Generation with Haiku**:

```python
import anthropic

client = anthropic.Anthropic()

SENTIMENT_PROMPT = """Rate the sentiment expressed toward {entity} in the following financial news text.

Score from -1.0 to +1.0:
- -1.0: Extremely negative (bankruptcy, fraud, severe losses, major lawsuits)
- -0.5: Moderately negative (earnings miss, downgrade, layoffs, headwinds)
-  0.0: Neutral/factual reporting (no clear positive or negative stance)
- +0.5: Moderately positive (earnings beat, upgrade, expansion, growth)
- +1.0: Extremely positive (exceptional performance, breakthrough, major wins)

Consider:
- Direct statements about the entity's performance, outlook, or actions
- Analyst opinions and ratings mentioned
- Comparisons to competitors or expectations
- Forward-looking statements and guidance

Text:
{text}

Entity: {entity}

Return ONLY a single decimal number between -1.0 and 1.0, nothing else."""

def generate_sentiment_label(text, entity):
    """Generate sentiment score using Claude Haiku."""
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": SENTIMENT_PROMPT.format(entity=entity, text=text)
        }]
    )
    score = float(response.content[0].text.strip())
    return max(-1.0, min(1.0, score))  # Clamp to valid range
```

**Batch Label Generation**:
```python
import json
from pathlib import Path
from tqdm import tqdm

def generate_training_labels(articles, output_path):
    """
    Generate Haiku sentiment labels for training data.
    
    Args:
        articles: List of {"text": str, "entities": List[str]}
        output_path: Where to save labeled data
    """
    labeled_data = []
    
    for article in tqdm(articles):
        text = article["text"]
        for entity in article["entities"]:
            try:
                score = generate_sentiment_label(text, entity)
                labeled_data.append({
                    "text": text,
                    "entity": entity,
                    "sentiment_score": score
                })
            except Exception as e:
                print(f"Error labeling {entity}: {e}")
                continue
    
    with open(output_path, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    return labeled_data
```

**Sentiment Head Architecture**:
```python
class SentimentHead(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8):
        super().__init__()
        
        # Entity-focused attention pooling
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Regression layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
    
    def forward(self, encoder_output, entity_position_mask):
        """
        Args:
            encoder_output: (batch, seq_len, hidden_size) from Longformer
            entity_position_mask: (batch, seq_len) binary mask, 1 for entity tokens
        
        Returns:
            sentiment_score: (batch,) continuous values in [-1, 1]
        """
        # Create attention mask: True = ignore, so invert entity mask
        key_padding_mask = (entity_position_mask == 0)
        
        # Self-attention focusing on entity positions
        attn_output, attn_weights = self.entity_attention(
            query=encoder_output,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=key_padding_mask
        )
        
        # Pool entity token representations
        # Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, hidden_size)
        mask_expanded = entity_position_mask.unsqueeze(-1).expand_as(attn_output)
        
        # Sum entity token outputs
        summed = torch.sum(attn_output * mask_expanded, dim=1)
        
        # Normalize by entity token count
        num_entity_tokens = entity_position_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        pooled = summed / num_entity_tokens
        
        # Regression to scalar
        x = self.dropout(pooled)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        score = self.fc2(x).squeeze(-1)
        
        # Bound output to [-1, 1] using tanh
        return torch.tanh(score)
```

**Training Loop**:
```python
def train_sentiment_head(model, train_loader, val_loader, epochs=10, lr=2e-5):
    """
    Train sentiment head with MSE loss on Haiku-generated labels.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            entity_mask = batch['entity_mask'].cuda()
            global_attention_mask = batch['global_attention_mask'].cuda()
            labels = batch['sentiment_score'].cuda()
            
            optimizer.zero_grad()
            
            # Forward pass through encoder
            encoder_output = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            ).last_hidden_state
            
            # Forward pass through sentiment head
            predictions = model.sentiment_head(encoder_output, entity_mask)
            
            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                entity_mask = batch['entity_mask'].cuda()
                global_attention_mask = batch['global_attention_mask'].cuda()
                labels = batch['sentiment_score'].cuda()
                
                encoder_output = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask
                ).last_hidden_state
                
                predictions = model.sentiment_head(encoder_output, entity_mask)
                val_loss += criterion(predictions, labels).item()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
```

---

## Complete Model Implementation

```python
import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from fastcoref import FCoref
from typing import List, Dict, Tuple, Optional

class FinancialEntitySentimentModel(nn.Module):
    """
    Complete pipeline for entity-level sentiment analysis on financial news.
    
    Components:
    - Longformer encoder (shared)
    - NER head (optional, for entity detection)
    - Coreference resolution (FastCoref)
    - Sentiment regression head (trained on Haiku labels)
    """
    
    def __init__(
        self,
        encoder_name: str = "allenai/longformer-base-4096",
        hidden_size: int = 768,
        num_attention_heads: int = 8,
        num_ner_labels: int = 15,
        use_ner_head: bool = True,
        device: str = "cuda:0"
    ):
        super().__init__()
        
        self.device = device
        self.use_ner_head = use_ner_head
        
        # Shared encoder
        self.encoder = LongformerModel.from_pretrained(encoder_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(encoder_name)
        
        # NER head (optional)
        if use_ner_head:
            self.ner_head = NERHead(hidden_size, num_ner_labels)
        
        # Coreference (external model, not nn.Module)
        self.coref_model = FCoref(device=device)
        
        # Sentiment head
        self.sentiment_head = SentimentHead(hidden_size, num_attention_heads)
        
        # Ticker-company mapping for coreference augmentation
        self.ticker_map = {}
    
    def load_ticker_map(self, path: str):
        """Load ticker to company name mappings."""
        import json
        with open(path) as f:
            self.ticker_map = json.load(f)
    
    def _find_entity_positions(
        self,
        input_ids: torch.Tensor,
        entity_tokens: List[int]
    ) -> List[Tuple[int, int]]:
        """Find all occurrences of entity tokens in input."""
        positions = []
        input_list = input_ids.tolist()
        entity_len = len(entity_tokens)
        
        for i in range(len(input_list) - entity_len + 1):
            if input_list[i:i + entity_len] == entity_tokens:
                positions.append((i, i + entity_len))
        
        return positions
    
    def _create_entity_mask(
        self,
        seq_len: int,
        positions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Create binary mask for entity positions."""
        mask = torch.zeros(seq_len, dtype=torch.float32)
        for start, end in positions:
            mask[start:end] = 1.0
        return mask
    
    def _expand_positions_with_coref(
        self,
        text: str,
        entity: str,
        base_positions: List[Tuple[int, int]],
        encoding
    ) -> List[Tuple[int, int]]:
        """Expand entity positions with coreference mentions."""
        # Get coreference clusters
        coref_clusters = self.coref_model.predict(texts=[text])[0].get_clusters()
        
        # Find which cluster contains our entity
        entity_cluster = None
        for cluster in coref_clusters:
            for mention in cluster:
                if entity.lower() in mention.lower() or mention.lower() in entity.lower():
                    entity_cluster = cluster
                    break
            if entity_cluster:
                break
        
        if not entity_cluster:
            return base_positions
        
        # Add ticker mapping expansions
        expanded_mentions = set(entity_cluster)
        if entity.upper() in self.ticker_map:
            expanded_mentions.update(self.ticker_map[entity.upper()])
        for ticker, names in self.ticker_map.items():
            if entity in names:
                expanded_mentions.add(ticker)
                expanded_mentions.update(names)
        
        # Find token positions for all coreferent mentions
        all_positions = list(base_positions)
        for mention in expanded_mentions:
            if mention == entity:
                continue
            mention_tokens = self.tokenizer.encode(mention, add_special_tokens=False)
            positions = self._find_entity_positions(encoding.input_ids[0], mention_tokens)
            all_positions.extend(positions)
        
        # Remove duplicates and sort
        all_positions = list(set(all_positions))
        all_positions.sort(key=lambda x: x[0])
        
        return all_positions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_mask: torch.Tensor,
        global_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            entity_mask: (batch, seq_len) binary mask for entity positions
            global_attention_mask: (batch, seq_len) for Longformer
        
        Returns:
            sentiment_scores: (batch,) continuous [-1, 1]
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        ).last_hidden_state
        
        sentiment_scores = self.sentiment_head(encoder_output, entity_mask)
        
        return sentiment_scores
    
    @torch.no_grad()
    def analyze(
        self,
        text: str,
        target_entities: Optional[List[str]] = None,
        use_coref: bool = True
    ) -> Dict:
        """
        Full inference pipeline for a single article.
        
        Args:
            text: Article text
            target_entities: List of entities to analyze (if None, uses NER)
            use_coref: Whether to expand entities with coreference
        
        Returns:
            Dict with entity-level sentiment scores
        """
        self.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding="max_length"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Get entities (from NER head or provided)
        if target_entities is None and self.use_ner_head:
            # Run NER to detect entities
            encoder_output = self.encoder(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            ).last_hidden_state
            ner_logits = self.ner_head(encoder_output)
            # Parse NER predictions to get entity spans
            # (simplified - would need proper BIO decoding)
            target_entities = self._decode_ner(ner_logits, encoding)
        
        if not target_entities:
            return {"error": "No entities provided or detected"}
        
        results = {}
        
        for entity in target_entities:
            # Find base entity positions
            entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False)
            base_positions = self._find_entity_positions(
                encoding['input_ids'][0],
                entity_tokens
            )
            
            if not base_positions:
                results[entity] = {
                    "sentiment_score": None,
                    "error": "Entity not found in text"
                }
                continue
            
            # Expand with coreference
            if use_coref:
                all_positions = self._expand_positions_with_coref(
                    text, entity, base_positions, encoding
                )
            else:
                all_positions = base_positions
            
            # Create masks
            seq_len = encoding['input_ids'].shape[1]
            entity_mask = self._create_entity_mask(seq_len, all_positions)
            entity_mask = entity_mask.unsqueeze(0).to(self.device)
            
            global_attention_mask = torch.zeros_like(encoding['input_ids'])
            global_attention_mask[:, 0] = 1  # CLS
            for start, end in all_positions:
                global_attention_mask[:, start:end] = 1
            
            # Forward pass
            encoder_output = self.encoder(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                global_attention_mask=global_attention_mask
            ).last_hidden_state
            
            sentiment_score = self.sentiment_head(encoder_output, entity_mask)
            
            results[entity] = {
                "sentiment_score": sentiment_score.item(),
                "mentions": all_positions,
                "num_mentions": len(all_positions)
            }
        
        return results
    
    def _decode_ner(self, logits, encoding):
        """Decode NER logits to entity strings (simplified)."""
        # Placeholder - implement proper BIO decoding
        predictions = logits.argmax(dim=-1)
        # Would extract spans and convert back to strings
        return []


class NERHead(nn.Module):
    """Token classification head for named entity recognition."""
    
    def __init__(self, hidden_size: int = 768, num_labels: int = 15):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(encoder_output)
        return self.classifier(x)


class SentimentHead(nn.Module):
    """
    Entity-focused sentiment regression head.
    Uses attention pooling over entity positions to produce continuous score.
    """
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 8):
        super().__init__()
        
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        entity_position_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, hidden_size)
            entity_position_mask: (batch, seq_len) with 1s at entity positions
        
        Returns:
            sentiment_score: (batch,) in [-1, 1]
        """
        key_padding_mask = (entity_position_mask == 0)
        
        attn_output, _ = self.entity_attention(
            query=encoder_output,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=key_padding_mask
        )
        
        mask_expanded = entity_position_mask.unsqueeze(-1).expand_as(attn_output)
        summed = torch.sum(attn_output * mask_expanded, dim=1)
        num_tokens = entity_position_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        pooled = summed / num_tokens
        
        x = self.dropout(pooled)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        score = self.fc2(x).squeeze(-1)
        
        return torch.tanh(score)
```

---

## Data Pipeline

### Dataset Class

```python
import torch
from torch.utils.data import Dataset
import json

class SentimentDataset(Dataset):
    """Dataset for training sentiment head with Haiku-generated labels."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entity = item['entity']
        sentiment_score = item['sentiment_score']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Find entity positions
        entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False)
        input_ids = encoding['input_ids'].squeeze(0)
        
        entity_mask = torch.zeros(self.max_length, dtype=torch.float32)
        global_attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        global_attention_mask[0] = 1  # CLS token
        
        # Find all entity occurrences
        input_list = input_ids.tolist()
        entity_len = len(entity_tokens)
        
        for i in range(len(input_list) - entity_len + 1):
            if input_list[i:i + entity_len] == entity_tokens:
                entity_mask[i:i + entity_len] = 1.0
                global_attention_mask[i:i + entity_len] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'entity_mask': entity_mask,
            'global_attention_mask': global_attention_mask,
            'sentiment_score': torch.tensor(sentiment_score, dtype=torch.float32)
        }
```

### Training Script

```python
import torch
from torch.utils.data import DataLoader, random_split
from transformers import LongformerTokenizer

def main():
    # Config
    DATA_PATH = "haiku_labeled_data.json"
    MODEL_SAVE_PATH = "financial_sentiment_model"
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    VAL_SPLIT = 0.1
    
    # Initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    # Load dataset
    full_dataset = SentimentDataset(DATA_PATH, tokenizer)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = FinancialEntitySentimentModel(device=str(device))
    model.to(device)
    
    # Train
    train_sentiment_head(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Save
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
```

---

## Inference Optimization

### torch.compile (PyTorch 2.0+)

```python
model = FinancialEntitySentimentModel()
model = torch.compile(model, mode="reduce-overhead")
```

### Mixed Precision

```python
from torch.cuda.amp import autocast

@torch.no_grad()
def fast_inference(model, text, entities):
    with autocast(dtype=torch.float16):
        return model.analyze(text, entities)
```

### Batch Processing

```python
def batch_analyze(model, articles: List[Dict], batch_size: int = 16):
    """
    Process multiple articles efficiently.
    
    Args:
        articles: List of {"text": str, "entities": List[str]}
        batch_size: Articles per batch
    
    Returns:
        List of result dicts
    """
    results = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for article in batch:
            result = model.analyze(
                article["text"],
                article["entities"],
                use_coref=True
            )
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results
```

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Inference latency | < 500ms | Single article, A6000 |
| Batch throughput | > 50 articles/sec | Batch size 16, A6000 |
| Sentiment MAE | < 0.15 | vs Haiku labels on held-out set |
| Sentiment correlation | > 0.85 | Pearson r vs Haiku labels |

---

## File Structure

```
financial_entity_sentiment/
├── config/
│   └── model_config.yaml
├── data/
│   ├── raw/                      # Raw Bloomberg articles
│   ├── labeled/                  # Haiku-labeled training data
│   └── ticker_map.json           # Ticker to company mappings
├── models/
│   ├── encoder.py                # Longformer wrapper
│   ├── ner_head.py               # NER head
│   ├── coref_head.py             # FastCoref wrapper
│   ├── sentiment_head.py         # Sentiment regression head
│   └── pipeline.py               # Full model class
├── training/
│   ├── dataset.py                # PyTorch dataset
│   ├── train.py                  # Training loop
│   └── label_generator.py        # Haiku label generation
├── inference/
│   ├── predictor.py              # Inference wrapper
│   └── optimize.py               # TensorRT/ONNX export
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## Requirements

```
torch>=2.0.0
transformers>=4.36.0
fastcoref>=0.2.0
anthropic>=0.18.0
accelerate>=0.25.0
sentencepiece
protobuf
tqdm
```

---

## Quick Start

```python
from financial_entity_sentiment import FinancialEntitySentimentModel

# Load trained model
model = FinancialEntitySentimentModel.from_pretrained("./trained_model")
model.cuda()
model.eval()

# Analyze article
result = model.analyze(
    text="Apple Inc. reported record quarterly revenue of $123.9 billion...",
    target_entities=["Apple Inc.", "Tim Cook"]
)

print(result)
# {
#     "Apple Inc.": {"sentiment_score": 0.72, "mentions": [(0, 10), (156, 166)], "num_mentions": 2},
#     "Tim Cook": {"sentiment_score": 0.45, "mentions": [(89, 97)], "num_mentions": 1}
# }
```
