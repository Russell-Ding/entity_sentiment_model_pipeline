# Label-noise audit — control vs gold (DeepSeek/Kimi) on the test split

Covered entities: 8,078. Buckets are the worst disagreements; adjudicate each: is the **model** wrong (model limit) or the **label** wrong (training noise)?
**Disagreement rates:** SIGN_FLIP 1.6%, MODEL_HOT 0.7%, MODEL_FLAT 3.7%, BIG_GAP 1.6%



## SIGN_FLIP — opposite signs, both strong  (127 total, 1.6%)
- **WMT** (ORG) gold=-0.5 pred=0.79 gap=1.29  [d806b9a01d53]
  _…es on retail optimism around holiday season. (For a live blog on the U.S. stock market, click or type LIVE/ in a news window.)  * Walmart hikes FY guidance, but Q3 margins hurt  * Home Depot hits record high on strong Q3  * October retail sales rise more than …_
- **CELH** (ORG) gold=0.6 pred=-0.45 gap=-1.05  [a83b6d479cba]
  _…ocks.  Among others, global economy bellwether Caterpillar lost 2.6% after raising its annual estimate for tariff-related costs.  Celsius Holdings rose 6.6% after a report said PepsiCo was increasing its stake in the energy drink maker through a $585 million d…_
- **GCI** (ORG) gold=0.3 pred=-0.31 gap=-0.61  [820782e2d1a9]
  _Gannett sues Google, alleges online ad monopoly. NEW YORK (Reuters) -Gannett Co Inc, the largest U.S. newspaper chain, on Tuesday sued Google, accusing the social media company of violating federal antitrust law by trying to monopolize the market for online ad…_
- **AAPL** (ORG) gold=0.5 pred=-0.51 gap=-1.01  [b82db04ff2a8]
  _UPDATE 5-Apple wins bid to pause Apple Watch ban at US appeals court. (Adds analyst comments, paragraphs 4, 8-9, store comment 15)  By Blake Brittain and Jaspreet Singh  Dec 27 (Reuters) - Apple scored a victory on Wednesday when a U.S. appeals court paused a …_
- **Rohit Prasad** (PERSON) gold=-0.4 pred=0.39 gap=0.79  [2fad594a740b]
  _…els, chip programs such as ⁠Graviton and Trainium, ​and emerging quantum computing ​initiatives.  The company also announced that Rohit Prasad, who helped build ‍Alexa and ⁠recently led development of Amazon's Nova foundation models, will leave ⁠at the end of …_
- **GOOGL** (ORG) gold=0.6 pred=-0.57 gap=-1.17  [757da77caa60]
  _UPDATE 1-U.S. Supreme Court sides with Google in major copyright dispute with Oracle. (Adds background to case, paragraphs 3-8)  By Andrew Chung  April 5 (Reuters) - The U.S. Supreme Court handed Alphabet Inc's Google a major victory on Monday, ruling that its…_

## MODEL_HOT — model strong, label ~0 (model sees stake the label calls incidental)  (55 total, 0.7%)
- **GOOGL** (ORG) gold=0.0 pred=-0.66 gap=-0.66  [d4a9f835a795]
  _…ices on their sites to users' acceptance of profiling cookies.  WHAT'S NEXT?  Italy has actively pursued tech companies over tax. Google in February agreed to pay 326 million euros to settle a tax claim relating to the period between 2015 and 2019.  Story Cont…_
- **TM** (ORG) gold=0.0 pred=0.52 gap=0.52  [ce415e50810c]
  _…tech offers improved energy density, lower costs and improved driving range. Partnerships with auto biggies like Tesla and Toyota TM are likely to boost the firm’s prospects. Notably, the company is targeting zero cobalt in its battery cells and plans to comme…_
- **Mark Zuckerberg** (PERSON) gold=0.0 pred=-0.57 gap=-0.57  [ee59954e3af4]
  _…planned to lay off about 10,000 employees, or roughly 13% of its workforce, the latest move to hew to what the company's founder, Mark Zuckerberg, has called a "year of efficiency".  - Boeing said on Tuesday that it had secured orders for dozens of 787 Dreamli…_
- **NVDA** (ORG) gold=0.0 pred=0.64 gap=0.64  [e26b8a77bb85]
  _Super Micro Computer’s 700% Gain Eclipses Nvidia as ‘Backdoor’ to AI Frenzy. (Bloomberg) -- For more than a year, Nvidia Corp. has been the go-to trade for investors seeking exposure to the red-hot growth story of artificial intelligence. And yet there’s a low…_
- **CITADEL** (ORG) gold=0.0 pred=0.64 gap=0.64  [08cc6d1b4aa3]
  _Ken Griffin’s Citadel Plans to Open a Palm Beach Office as Part of Florida Move. (Bloomberg) -- Ken Griffin’s Citadel plans to open a Palm Beach office, taking over the former Neiman Marcus department store building on the Florida town’s main shopping street. …_
- **AAPL** (ORG) gold=0.0 pred=-0.76 gap=-0.76  [520ae7c00bad]
  _…awmakers' data -New York Times. WASHINGTON (Reuters) - The U.S. Justice Department under former President Donald Trump subpoenaed Apple Inc for data from the accounts of at least two Democrats on the House of Representatives Intelligence Committee in an attemp…_

## MODEL_FLAT — label strong, model ~0 (model misses sentiment the label has)  (297 total, 3.7%)
- **SEA_LTD** (ORG) gold=-0.6 pred=-0.05 gap=0.55  [2e5da664a437]
  _…utting jobs and shutting parts of their operations to shore up balance sheets ahead of a potential recession.  In Southeast Asia, Sea Ltd. and Grab Holdings Ltd., Singapore’s biggest tech companies, are emblematic of this new reality: Their US-traded stocks ha…_
- **ARM** (ORG) gold=0.5 pred=0.09 gap=-0.41  [e3575620c83c]
  _…ropean stock markets, click or type LIVE/ in a news window.)  *  Cisco to cut over 4,000 jobs, lowers annual revenue forecast  *  Arm, SoundHound AI shares jump as Nvidia builds stake  *  Deere cuts 2024 profit view  *  Albemarle shares down on Q4 loss  *  Fut…_
- **NOW** (ORG) gold=0.7 pred=0.01 gap=-0.69  [4c685c105b41]
  _…touched the surface for AI demand and use cases, so investors need to be patient and play the long game.”  Tech Chart of the Day  ServiceNow surged 22% last week, capping a record weekly gain, after the software company issued an outlook for sales growth that …_
- **HUAWEI** (ORG) gold=0.5 pred=0.05 gap=-0.45  [68516231e4a3]
  _… for its lack of artificial intelligence features in China, a challenge for the U.S. giant as it battles growing competition from Huawei Technologies in the world's largest smartphone market.  Apple unveiled its long-awaited, AI-boosted iPhone 16 on Monday, ho…_
- **STZ** (ORG) gold=0.5 pred=-0.02 gap=-0.52  [c7c26a25b1ad]
  _… cheeky? How Europe’s best investor picks stocks including GE Aerospace and Microsoft  Berkshire’s biggest buy of the quarter was Constellation Brands Inc. STZ, acquiring 6,384,676 shares, increasing its holdings by more than 113% to a total of 12,009,000 shar…_
- **NTDOY** (ORG) gold=0.5 pred=-0.0 gap=-0.5  [1e36f8e8008a]
  _…lligence chip market, failed to impress investors with its revenue forecast after an eye-popping rally sent expectations soaring. Nintendo Co.’s stock rose by its most in six months after the company raised its Switch 2 outlook, a strong signal of confidence i…_

## BIG_GAP — |pred-gold| >= 0.6  (132 total, 1.6%)
- **BOLSONARO** (PERSON) gold=-0.6 pred=0.13 gap=0.73  [a3626dfabdd6]
  _…o Bolsonaro to Save the Amazon?. (Bloomberg Opinion) -- Investors with more than $4.5 trillion in assets want Brazilian President Jair Bolsonaro to stop loosening environmental rules and do more to control escalating deforestation in the Amazon and beyond. Thi…_
- **AAPL** (ORG) gold=0.4 pred=-0.23 gap=-0.63  [13c7d8bac72a]
  _… after earnings from Amazon (NASDAQ:AMZN) failed to live up to lofty expectations, sending its shares tumbling 6.6% after hours.  Apple (O:AAPL), meanwhile, forecast revenue well above Wall Street’s estimates, following strong June-quarter results supported by…_
- **MSFT** (ORG) gold=0.1 pred=-0.57 gap=-0.67  [cbecc9a6f2c3]
  _…osoft threatens to restrict data from rival AI search tools - Bloomberg News. (Adds details from Bloomberg)  March 24 (Reuters) - Microsoft Corp has threatened to cut off access to its internet-search data, which it licenses to rival search engines, if they do…_
- **SUNR** (ORG) gold=0.2 pred=0.9 gap=0.7  [9a9cbbdce1f8]
  _Tesla shows interest in Sunrise New Energy's battery components. (Reuters) - Battery components maker Sunrise New Energy said on Wednesday that it had received interest for its products from electric vehicle maker Tesla Inc.  Sunrise shares rose as much as 11%…_
- **AMZN** (ORG) gold=-0.4 pred=0.22 gap=0.62  [9e43af9243fb]
  _…34%, at 3,914.80.  The Nasdaq Composite was down 37.56 points, or 0.33%, at 11,317.06, dragged down by Tesla Inc, Nvidia Corp and Amazon.com.  Cloud service provider VMWare Inc surged 16.9% after reports over the weekend said chipmaker Broadcom Inc was in talk…_
- **Branch Metrics** (ORG) gold=0.2 pred=-0.7 gap=-0.9  [20ed50b3adf1]
  _Google created hurdles to protect smartphone foothold -small search firm. By Diane Bartz  WASHINGTON (Reuters) - The founder of Branch Metrics, which developed a method of searching within smartphone apps, told a U.S. antitrust trial on Wednesday how his compa…_
