=== Part E: recall measurement on 300k train S1 sample vs full train pools ===
Sample S1 entities: 300000
True (S1,match) pairs among sample: 1037463, across 283216 non-singleton S1 entities

Pair recall BEFORE top-K cut (per key type, and union):
  K1: 779404/1037463 (75.13%)
  K2: 392494/1037463 (37.83%)
  K2_nostate: 28/1037463 (0.00%)
  K3_exact: 481703/1037463 (46.43%)
  K3_pfx8: 376882/1037463 (36.33%)
  K4_exact: 9962/1037463 (0.96%)
  K4_state: 18050/1037463 (1.74%)
  K5: 250890/1037463 (24.18%)
  K5_nostate: 14/1037463 (0.00%)
  K6: 465926/1037463 (44.91%)
  UNION (any key): 948390/1037463 (91.41%)

Union recall BEFORE cut, by country:
  US: 581588/622464 (93.43%)
  India: 366802/414999 (88.39%)

Union recall BEFORE cut, by match source:
  S3: 491023/535974 (91.61%)
  S2: 457367/501489 (91.20%)

Union recall AFTER top-K cut, swept over K:
  K=10: 912814/1037463 (87.99%)
  K=20: 932804/1037463 (89.91%)
  K=30: 940110/1037463 (90.62%)
  K=40: 943883/1037463 (90.98%)
  K=60: 947263/1037463 (91.31%)

No swept K reached 97% union recall; largest K=60 achieved only 91.31%. See 'what I'd try next' in the report.

At K=60: non-singleton S1 with ALL true matches found: 221787/283216 (78.31%); with >=1 found: 278637/283216 (98.38%)

Candidates per S1 at K=60: mean=34.0, median=34, p95=60, max=60, total candidate pairs=10156253

30 random MISSED true pairs (never matched by any key), out of 89073 total misses:
  [no shared number]
    S1-559054648: 'Apex Asset Worldwide Inc' | '2587 Iowa Avenue, Ogden City, UT'  (name_core='apex asset worldwide', addr_words='ogden ut', numbers='2587', state='ia')
    S2-438185829: 'Apex Worldwide Inc Service' | ''  (name_core='apex worldwide service', addr_words='', numbers='', state='')
  [no shared number]
    S1-182509031: 'Back Alley Seafood L.L.C.' | '941 Pine Avenue, Rialto, CA'  (name_core='back alley seafood', addr_words='pine rialto', numbers='941', state='ca')
    S2-923479143: 'Back-Alley Seafood' | 'PINE AVENUE, RIALTO, CA'  (name_core='back alley seafood', addr_words='pine rialto', numbers='', state='ca')
  [no shared number]
    S1-363467039: 'Lake Continental Spade LLC' | '3533 Chewning Road, Oxford, NC'  (name_core='lake continental spade', addr_words='chewning oxford', numbers='3533', state='nc')
    S3-114470048: 'Lake Colfnitnental Spade LLC' | ''  (name_core='lake colfnitnental spade', addr_words='', numbers='', state='')
  [no shared number]
    S1-476185589: 'Guru Finance Private Limited' | 'At - Hetagadi Po - Kumbhiragadia, Via - Danagadi, Kalinganagar, Jajpur, Orissa'  (name_core='guru finance private', addr_words='at hetagadi po kumbhiragadia via danagadi kalinganagar jajpur', numbers='', state='od')
    S3-152215628: 'Guru Finance Limited  Services' | 'At - Hetagadi Po - Kumbhiragadia, Via - Danagadi, Kalinganagar, Jajpur, ଓଡ଼ିଶା'  (name_core='guru finance services', addr_words='at hetagadi po kumbhiragadia via danagadi kalinganagar jajpur odd ishaa', numbers='', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-798083637: 'Golden Logistics Private Limited' | '301, 3Rd Floor, Nitco Biz Park, Plot No. C/19, Road No.16/U, Wagle Estate, Thane West, Thane, Maharashtra'  (name_core='golden logistics private', addr_words='3rd floor nitco biz park c u wagle estate thane west', numbers='3,16,19,301', state='mh')
    S3-748658466: 'Limited Golden Logistics Partners' | 'A-301, Thane West, Thane, महाराष्ट्र'  (name_core='logistics partners', addr_words='a thane west mhaaraassttr', numbers='301', state='')
  [no shared name_core token]
    S1-938317969: 'Gold Producer Private Limited' | '50, Udaipur, Shastri Circle Road, Rajasthan, Indraprasth Complex, Delhi Gate'  (name_core='gold producer private', addr_words='udaipur shastri indraprasth complex dl gate', numbers='50', state='rj')
    S2-733037390: 'गोल्ड प्रोड्यूसर प्राइवेट लिमिटेड' | '50, SHASTRI CIRCLE ROAD, UDAIPUR, राजस्थान'  (name_core='goldd proddyuusr praaivett limaaittedd', addr_words='shastri udaipur raajsthaan', numbers='50', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-177510579: 'India Raj Springs Group' | 'Syno 52/1B, C/O Vasanth M, Sampoorna Farms, Mysore, Karnataka'  (name_core='india raj springs group', addr_words='syno 1b c o vasanth m sampoorna farms mysore', numbers='1,52', state='ka')
    S3-215450369: 'India-Raj Srnigs Group' | 'Syno B3/52/1b, Mysore, KA'  (name_core='india raj srnigs group', addr_words='syno b3 1b mysore', numbers='1,3,52', state='ka')
  [no shared number]
    S1-343355142: 'Automated Textile Union LLC' | '12311 Silent Creek Drive, Pearland, TX'  (name_core='automated textile union', addr_words='silent creek pearland', numbers='12311', state='tx')
    S3-724370735: 'Automated Textile LLC Center' | '2311 Silent Creek Drive, Pearland, Texas'  (name_core='automated textile center', addr_words='silent creek pearland', numbers='2311', state='tx')
  [no shared name_core token]
    S1-869436801: 'Innovative Producer Private Limited' | 'Office No. 601, 6Th Floor, Sharada Terrace, Plot No. 65, Sector-11, Cbd Belapur, Navi Mumbai, Thane, Maharashtra'  (name_core='innovative producer private', addr_words='office 6th floor sharada sector cbd belapur navi mumbai thane', numbers='6,11,65,601', state='mh')
    S3-865720222: 'इनोवेटिव प्रोड्यूसर प्राइवेट लिमिटेड' | 'Office No. 601., Navi Mumbai Region, Thane, MH'  (name_core='inovettiv proddyuusr praaivett limaaittedd', addr_words='office navi mumbai region thane', numbers='601', state='mh')
  [no shared name_core token]
    S1-248784742: 'Golden Consultancy Private Limited' | 'Plot No247/5-6Nr Rajlaxmiprints G I D C Pandesar, Surat, Gujarat'  (name_core='golden consultancy private', addr_words='no247 6nr rajlaxmiprints g i d c pandesar surat', numbers='5,6,247', state='gj')
    S2-567966299: 'ગોલ્ડન કન્સલ્ટન્સી પ્રાઇવેટ લિમિટેડ' | 'NO 247/5-6NR RJLAXMIPRINTS G I D C PANDESAR, SURAT, ગુજરાત'  (name_core='golddn knslttnsii praaivett limittedd', addr_words='6nr rjlaxmiprints g i d c pandesar surat gujraat', numbers='5,6,247', state='')
  [no shared number]
    S1-583293380: 'Metropolitan Alliance' | '27 E Castle Hill Road, Agawam, MA'  (name_core='metropolitan alliance', addr_words='e castle hill agawam', numbers='27', state='ma')
    S2-857607072: 'Metropolitan Alliance' | '#6 CASTLE HILL ROAD, AGAWAM, MA'  (name_core='metropolitan alliance', addr_words='castle hill agawam', numbers='6', state='ma')
  [no shared name token AND no shared number]
    S1-491284731: 'Downtown Cafe' | 'WA, 221 Woodin Avenue, Chelan, Unit UNIT'  (name_core='downtown cafe', addr_words='woodin chelan unit', numbers='221', state='wa')
    S2-396286836: 'WEXXYLOECTO' | 'CELAN CDP, WA, 64 WOODIN AVENUE'  (name_core='wexxyloecto', addr_words='celan cdp woodin', numbers='64', state='wa')
  [no shared number]
    S1-719337864: 'Trinity Zion' | '4312 13th Avenue, Chattanooga, TN'  (name_core='trinity zion', addr_words='13th chattanooga', numbers='13,4312', state='tn')
    S3-60959384: 'Trinity Zion Industries' | ''  (name_core='trinity zion industries', addr_words='', numbers='', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-97171905: 'Housing Trust' | '515 Main Street, Unit UNIT 510, Wichita, KS'  (name_core='housing trust', addr_words='main unit wichita', numbers='510,515', state='ks')
    S2-224917346: 'Housing Trust' | '00515 Main Street, NULL, WICHIAT, KS'  (name_core='housing trust', addr_words='main null wichiat', numbers='515', state='ks')
  [no shared number]
    S1-283972214: 'Global Traders Private Limited' | 'C/O. Saithalavi, President Masjidul Mubarak, Malamkunnu, Mysoore Mala Post, Kozhikode, Kerala'  (name_core='global traders private', addr_words='c o saithalavi president masjidul mubarak malamkunnu mysoore mala post kozhikode', numbers='', state='kl')
    S2-981040970: 'Limited Global Traders Center' | 'C/O. SAITHALAVI, CALICUT, KOZHIKODE, Kerala'  (name_core='traders center', addr_words='c o saithalavi calicut kozhikode', numbers='', state='kl')
  [no shared number]
    S1-459674078: 'New Management Private Limited' | 'Abhavats B03, Aditi Apartments, D-1, Janakpuri, West Delhi, Delhi'  (name_core='new management private', addr_words='abhavats b03 aditi apartments d janakpuri west', numbers='1,3', state='dl')
    S3-378509544: 'New Management Private Enterprises' | ''  (name_core='new management private enterprises', addr_words='', numbers='', state='')
  [no shared number]
    S1-438191291: 'Advanced Software Alliance, LLC' | '22 Pinnacle Mountain Road, Leicester, NC'  (name_core='advanced software alliance', addr_words='pinnacle mountain leicester', numbers='22', state='nc')
    S2-272866974: 'Advanced Software Alliance, Llc - 2900611582' | 'PINNACLE MOUNTAIN RD, LEICESTER, NC'  (name_core='advanced software alliance 2900611582', addr_words='pinnacle mountain leicester', numbers='', state='nc')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-857509371: 'One Energy Private Limited' | 'New No 6 (Old No 22), C P Ramasamy Road Alwarpet, Chennai, Tamil Nadu'  (name_core='one energy private', addr_words='new old c p ramasamy alwarpet chennai', numbers='6,22', state='tn')
    S3-971111858: 'Limited One Energy Center' | 'Door No C-6 (Old No 22), Chennai, தமிழ்நாடு'  (name_core='one energy center', addr_words='c old chennai tmilllnaattu', numbers='6,22', state='')
  [no shared number]
    S1-639208014: 'Global Financial Labs LLP' | '10 Haskell Road, Pepperell, MA'  (name_core='global financial labs llp', addr_words='haskell pepperell', numbers='10', state='ma')
    S2-948361412: 'Global Financial Labs' | ''  (name_core='global financial labs', addr_words='', numbers='', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-513496023: 'Jain Media Private Limited' | 'H. No. 8-B/1, Plot No. 1-A, Ground Floor, Dlf Industrial Area, Moti Nagar, Delhi, West Delhi, Delhi'  (name_core='jain media private', addr_words='h b a ground floor dlf industrial area moti nagar west', numbers='1,8', state='dl')
    S2-585532125: 'जैन मीडिया प्राइवेट लिमिटेड' | 'WEST DELHI, H. NO. 8-B/1/2, PLOT NO. 1-A, GROUND FLOOR, DLF INDUSTRIAL AREA, MOTI NAGAR, Delhi'  (name_core='jain maaiiddiyaa praaivett limaaittedd', addr_words='west h b a ground floor dlf industrial area moti nagar', numbers='1,2,8', state='dl')
  [no shared number]
    S1-27691292: 'Highland Charter School Inc' | '638 Windy Point Drive, Mount Vernon, TX'  (name_core='highland charter school', addr_words='windy point mount vernon', numbers='638', state='tx')
    S2-936261780: 'Highland  Charter School Inc' | '63 WINDY POINT DR, MOUNT VERNON, TX'  (name_core='highland charter school', addr_words='windy point mount vernon', numbers='63', state='tx')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-451199208: 'Jain Infrastructure Private Limited' | '84/83, Srinidhi Layout, Kambipura(V), Kumbalgodu, Bangalore South, Bangalore, Karnataka'  (name_core='jain infrastructure private', addr_words='srinidhi layout kambipura v kumbalgodu bangalore south', numbers='83,84', state='ka')
    S2-66151912: 'ಜೈನ್ ಇನ್\u200cಫ್ರಾಸ್ಟ್ರಕ್ಚರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್' | '84/83, BANGALORE, BANGALORE SOUTH, Karnataka'  (name_core='jain inphraasttrkcr praiveett limittedd', addr_words='bangalore south', numbers='83,84', state='ka')
  [no shared name token AND no shared number]
    S1-267323134: 'Housing League' | '2009 Liberty Loop, Spotsylvania County, VA'  (name_core='housing league', addr_words='liberty loop spotsylvania county', numbers='2009', state='va')
    S2-224168853: 'housingleague.com' | 'LIBERTY LOOP, SPOTSYLVANIA COUNTY, VA'  (name_core='housingleague com', addr_words='liberty loop spotsylvania county', numbers='', state='va')
  [no shared number]
    S1-948343544: 'Frontier Council' | '102 Main Street, Fairbank, IA'  (name_core='frontier council', addr_words='main fairbank', numbers='102', state='ia')
    S2-405188320: 'Frontier Council LLC' | '10 MAIN ST, FAIRBANK, IA'  (name_core='frontier council', addr_words='main fairbank', numbers='10', state='ia')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-970122662: 'Wildlife Network' | '300 3775, Vernal, UT'  (name_core='wildlife network', addr_words='vernal', numbers='300,3775', state='ut')
    S2-556725757: 'Wildlife Network Corp' | '3775, VERRNAL, UT'  (name_core='wildlife network', addr_words='verrnal', numbers='3775', state='ut')
  [no shared number]
    S1-221501942: 'A Apex Charles' | '776 Bronx River Road, Unit Apartment B36, Yonkers, NY'  (name_core='a apex charles', addr_words='bronx river unit apartment b36 yonkers', numbers='36,776', state='ny')
    S2-645940450: 'A Apex' | '77 BRONX RIVER ROAD, YONKERS, NY'  (name_core='a apex', addr_words='bronx river yonkers', numbers='77', state='ny')
  [no shared number]
    S1-684947865: 'Marrilee K. Simpson, D.O. of Knoxville' | '4416 Balraj Lane, Knoxville, TN'  (name_core='marrilee k simpson do of knoxville', addr_words='balraj knoxville', numbers='4416', state='tn')
    S2-871833878: 'Marri1ee K. Simpson D.O. of Knoxville' | 'BALRAJ LANE, KNOXVILLE, TN'  (name_core='marri1ee k simpson do of knoxville', addr_words='balraj knoxville', numbers='', state='tn')
  [no shared name token AND no shared number]
    S1-334395888: 'Southern Engineering Limited' | 'C/O. Jeevan Medical Stores, Mahatma Gandhi Chowk Taluka- Shirol, Nandani, Kolhapur, Maharashtra'  (name_core='southern engineering', addr_words='c o jeevan medical stores mahatma gandhi chowk taluka shirol nandani kolhapur', numbers='', state='mh')
    S2-25687128: 'सदर्न इंजीनियरिंग लिमिटेड' | 'C/O. JEEVAN MEDICAL STORES, MAHATMA GANDHI CHOWK TALUKA- SHIROL, NANDANI, KOLHAPUR, Maharashtra'  (name_core='sdrn injiiniyring limaaittedd', addr_words='c o jeevan medical stores mahatma gandhi chowk taluka shirol nandani kolhapur', numbers='', state='mh')
  [no shared number]
    S1-600251093: 'City Consultants Private Limited' | 'C/O Nandhanan. U.S, Uttuvally House, Amballur, Alagappanagar P.O, Thrissur, Kerala'  (name_core='city consultants private', addr_words='c o nandhanan u s uttuvally amballur alagappanagar p thrissur', numbers='', state='kl')
    S3-852122166: 'Private City Consultants Ltd' | 'C/o Nandhanan. U.s, Uttuvally House, Amballur, Alagappanagar P.o, Thrissur, KL'  (name_core='private city consultants', addr_words='c o nandhanan u s uttuvally amballur alagappanagar p thrissur', numbers='', state='kl')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-571517885: 'Hotel Software Private Limited' | 'C-51, First Floor, Block-C, South Extension, Part-Ii, New Delhi, South Delhi, Delhi'  (name_core='hotel software private', addr_words='c first floor block south extension part ii', numbers='51', state='dl')
    S2-357390084: '... Hotel Sóftware' | 'PLOT 882 C-51, FARIDABAD, Delhi'  (name_core='hotel software', addr_words='c faridabad', numbers='51,882', state='dl')