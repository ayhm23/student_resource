=== Part E: recall on the 300k train S1 sample (candidates from all-train-S1 blocking) ===
Caps: default=50, strict=15 ('K2_nostate', 'K4_nostate', 'K5_nostate', 'K6'); top_k per S1=40; S1 batch=35625
Sample S1 entities: 300000
True (S1,match) pairs among sample: 1037463, across 283216 non-singleton S1 entities

Pair recall BEFORE top-K cut (per key type, and union):
  K1: 779817/1037463 (75.17%)
  K2: 427832/1037463 (41.24%)
  K2_nostate: 39/1037463 (0.00%)
  K3_exact: 481683/1037463 (46.43%)
  K3_pfx8: 381557/1037463 (36.78%)
  K4_exact: 10150/1037463 (0.98%)
  K4_nostate: 0/1037463 (0.00%)
  K4_state: 19957/1037463 (1.92%)
  K5: 267437/1037463 (25.78%)
  K5_nostate: 14/1037463 (0.00%)
  K6: 388101/1037463 (37.41%)
  K7d: 44798/1037463 (4.32%)
  K7o: 15930/1037463 (1.54%)
  UNION (any key): 957364/1037463 (92.28%)

Union recall BEFORE cut, by country:
  India: 369883/414999 (89.13%)
  US: 587481/622464 (94.38%)

Union recall BEFORE cut, by match source:
  S3: 492681/535974 (91.92%)
  S2: 464683/501489 (92.66%)

Union recall AFTER top-K cut, swept over K:
  K=5: 838840/1037463 (80.85%)
  K=10: 920721/1037463 (88.75%)
  K=20: 940495/1037463 (90.65%)
  K=30: 947540/1037463 (91.33%)
  K=40: 950991/1037463 (91.67%)

No swept K reached 97% union recall; largest K=40 achieved only 91.67%. See 'what I'd try next' in the report.

At K=40: non-singleton S1 with ALL true matches found: 224010/283216 (79.10%); with >=1 found: 277799/283216 (98.09%)

Candidates per S1 at K=40: mean=33.1, median=40, p95=40, max=40, total candidate pairs=9848668

30 random MISSED true pairs (never matched by any key), out of 80099 total misses:
  [no shared number]
    S1-746623155: 'Board of Sanitation' | '378 William Street, Unit Apartment 2, Port Chester, NY'  (name_core='board of sanitation', addr_words='william unit apartment port chester', numbers='2,378', state='ny')
    S3-853103769: 'Board of Sanitation Corp' | ''  (name_core='board of sanitation', addr_words='', numbers='', state='')
  [no shared number]
    S1-75450193: 'Chiropractic Liberty Partners Care' | 'VA, 531 Long Branch Drive, Mecklenburg County'  (name_core='chiropractic liberty partners care', addr_words='long branch mecklenburg county', numbers='531', state='va')
    S2-679389923: 'Chiropractic-Liberty Partners Care Trust' | ''  (name_core='chiropractic liberty partners care trust', addr_words='', numbers='', state='')
  [no shared number]
    S1-110933140: 'Shakti Business Private Limited' | 'Hamir Singh Nagar Unnav Road, Datia, Madhya Pradesh'  (name_core='shakti business private', addr_words='hamir singh nagar unnav datia', numbers='', state='mp')
    S3-827058508: 'शक्ति बिजनेस Private Limited' | 'Hamir Singh Nagar Unnav Road, Datia, Shivpuri, MP'  (name_core='shakti business private', addr_words='hamir singh nagar unnav datia shivpuri', numbers='', state='mp')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-780891026: 'Happy Tattoo' | 'Tyler, TX, 17910 31'  (name_core='happy tattoo', addr_words='tyler', numbers='31,17910', state='tx')
    S2-544143495: 'Happy Tattoo Inc.' | '77910 31, TYLEER, TX'  (name_core='happy tattoo', addr_words='tyleer', numbers='31,77910', state='tx')
  [no shared number]
    S1-971597190: 'Galaxy Power Pvt Ltd' | 'Picnic Spot Road, Indira Nagar, Lucknow, Uttar Pradesh, 14 Manas Greens'  (name_core='galaxy power pvt', addr_words='picnic spot indira nagar lucknow manas greens', numbers='14', state='up')
    S3-553366011: 'Galaxy Powre Pvt Ltd' | 'Manas Greens, NULL, Lucknow, उत्तर प्रदेश'  (name_core='galaxy powre pvt', addr_words='manas greens null lucknow', numbers='', state='up')
  [no shared number]
    S1-20956240: 'Prakasam Care Limited' | '8-62, Jain Complex Near Old Sbh Bank, Paruchuru Road Prakasam District, Inkollu, Prakasam, Andhra Pradesh'  (name_core='prakasam care', addr_words='jain complex near old sbh bank paruchuru prakasam district inkollu', numbers='8,62,862', state='ap')
    S2-350941202: 'Prakasam Cbe Límited' | ''  (name_core='prakasam cbe', addr_words='', numbers='', state='')
  [no shared number]
    S1-315918209: 'Patriot Resources LP' | '18737 250, Patriot, IN'  (name_core='patriot resources', addr_words='patriot', numbers='250,18737', state='in')
    S3-638689273: 'Patriot Résources LP' | ''  (name_core='patriot resources', addr_words='', numbers='', state='')
  [no shared number]
    S1-907501667: 'Pediatric Dentistry Modern Physicians LLC' | '298 Main Street, Elmo, UT'  (name_core='pediatric dentistry modern physicians', addr_words='main elmo', numbers='298', state='ut')
    S2-774617234: 'Pediatric Dentistry Modern' | ''  (name_core='pediatric dentistry modern', addr_words='', numbers='', state='')
  [no shared number]
    S1-789237518: 'Premier Ventures Private Limited' | 'Vpo Nirjan, Safidon Road Near Byepass, Jind, Haryana'  (name_core='premier ventures private', addr_words='vpo nirjan safidon near byepass jind', numbers='', state='hr')
    S3-542494716: 'प्रीमियर वेंचर्स प्राइवेट लिमिटेड' | 'Vpo Nirjan, Safidon Road Near Byepass, Jind, HR'  (name_core='premier ventures private', addr_words='vpo nirjan safidon near byepass jind', numbers='', state='hr')
  [no shared number]
    S1-337644348: 'Dermatology Grand Care LLC' | '22047 Gray Road, Dennis, KS'  (name_core='dermatology grand care', addr_words='gray dennis', numbers='22047', state='ks')
    S2-882670625: 'Dermatology Care Grand LLC' | 'GRAY RD, DENNIS, KS'  (name_core='dermatology care grand', addr_words='gray dennis', numbers='', state='ks')
  [no shared name_core token]
    S1-983175447: 'Veloce Future Pvt Ltd' | '103, Surya Samanpura Enclave I.G. Colony, Patna, Bihar'  (name_core='veloce future pvt', addr_words='surya samanpura enclave ig colony patna', numbers='103', state='br')
    S3-117040826: 'fveloce.com' | 'Patna, BR, Patna, 103'  (name_core='fveloce com', addr_words='patna', numbers='103', state='br')
  [no shared number]
    S1-377468446: 'Jay Marketing Pvt Ltd' | 'C/O Shailendra Srivastav, New Brahmani Tola, Fatehpur, Barabanki, Uttar Pradesh'  (name_core='jay marketing pvt', addr_words='c o shailendra srivastav new brahmani tola fatehpur barabanki', numbers='', state='up')
    S2-221274114: 'जय मार्केटिंग प्रा. लि.' | 'BARABANKI, FATEHPUR, NEW BRAHMANI TOLA, C/O SHAILENDRA SRIVASTAV, Uttar Pradesh'  (name_core='jy marketing pvt', addr_words='barabanki fatehpur new brahmani tola c o shailendra srivastav', numbers='', state='up')
  [no shared number]
    S1-740915778: 'Cure Ice Cream!' | '157 Simmons Trail, Royal, AR'  (name_core='cure ice cream', addr_words='simmons royal', numbers='157', state='ar')
    S3-129114567: 'Cure Ice Trading' | ''  (name_core='cure ice trading', addr_words='', numbers='', state='')
  [no shared number]
    S1-461353427: 'Electricians Local 937' | '25 Hoornkill Avenue, Lewes, DE'  (name_core='electricians local 937', addr_words='hoornkill lewes', numbers='25', state='de')
    S3-451860888: 'Electricians 937 Local' | ''  (name_core='electricians 937 local', addr_words='', numbers='', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-756743480: 'Eye Empire Center' | '10430 64, Bldg Building F, Daphne, AL'  (name_core='eye empire center', addr_words='bldg building f daphne', numbers='64,10430', state='al')
    S2-350834701: 'CENTER EMPIRE EYE' | '10430 64, DAPPHNE, AL'  (name_core='center empire eye', addr_words='dapphne', numbers='64,10430', state='al')
  [no shared number]
    S1-27701744: 'Anand Infra Private Limited' | 'C/O-Ramesh Chand Sharma Khedi, Teh. Toda Bhim, Karauli, Rajasthan'  (name_core='anand infra private', addr_words='c o ramesh chand sharma khedi teh toda bhim karauli', numbers='', state='rj')
    S3-74145931: 'आनंद इंफ्रा प्राइवेट लिमिटेड' | 'C/o-ramesh Chand Sharma Khedi, Teh. Toda Bhim, Karauli, RJ'  (name_core='anand infra private', addr_words='c o ramesh chand sharma khedi teh toda bhim karauli', numbers='', state='rj')
  [no shared number]
    S1-591788863: 'Syndicate & Brothers LLP' | 'Maharashtra, Mumbai, 214 Link Way Estatenew Link Road Malad (West), Mumbai'  (name_core='syndicate and brothers llp', addr_words='mumbai link way estatenew malad west', numbers='214', state='mh')
    S2-809966899: 'Syndicate & Bodters LLP' | ''  (name_core='syndicate and bodters llp', addr_words='', numbers='', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-836480906: 'Shyam Builders Private Limited' | 'No 3, 3/1, Annamalai Nagar, Karur Bye Pass Road, Tiruchirappalli, Tamil Nadu'  (name_core='shyam builders private', addr_words='annamalai nagar karur bye pass tiruchirappalli', numbers='1,3', state='tn')
    S2-885688408: 'ஷ்யாம் பில்டர்ஸ் பிரைவேட் லிமிடெட்' | 'H.NO 3, CENTRAL REGION TRICHIRAPALLI, TIRUCHIRAPPALLI, Tamil Nadu'  (name_core='shyam builders private', addr_words='h central region trichirapalli tiruchirappalli', numbers='3', state='tn')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-538438975: 'First Power' | 'Nagpur, Pandurang Mangal Karyalaya Godhani Road, Zingabai Takli, Maharashtra, 1St Floor, Nagpur'  (name_core='first power', addr_words='nagpur pandurang mangal karyalaya godhani zingabai takli 1st floor', numbers='1', state='mh')
    S3-882303306: 'First Power [Corporation]' | 'MH, Nagpur, 1St Floor, Nagpur'  (name_core='first power', addr_words='nagpur 1st floor', numbers='1', state='mh')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-876968840: 'Galaxy Developers Private Limited' | '18/11/1 18/11/4 Shop-615, Spot 18 Sai Chowk, Pune City, Pune, Maharashtra'  (name_core='galaxy developers private', addr_words='spot sai chowk pune', numbers='1,4,11,18,615', state='mh')
    S2-916006928: 'गैलेक्सी डेवलपर्स प्राइवेट लिमिटेड' | 'A-18/11/1 18/11/4 SHOP-615, PUNE CITY, PUNE, महाराष्ट्र'  (name_core='galaxy developers private', addr_words='a pune', numbers='1,4,11,18,615', state='mh')
  [no shared number]
    S1-22020222: 'Zevue Consultants Inc' | '291 Jackson Avenue, Spring City, TN'  (name_core='zevue consultants', addr_words='jackson spring', numbers='291', state='tn')
    S2-226239349: 'Zevue Cocnsoultnts Inc' | ''  (name_core='zevue cocnsoultnts', addr_words='', numbers='', state='')
  [no shared number]
    S1-898013613: 'Veterans Project Inc' | '47173 2, Malta, MT'  (name_core='veterans project', addr_words='malta', numbers='2,47173', state='mt')
    S2-347249325: 'veterans project inc' | ''  (name_core='veterans project', addr_words='', numbers='', state='')
  [no shared number]
    S1-992040404: 'Royal Textiles, Inc' | '366 Apricot Sun Way, Zebulon, NC'  (name_core='royal textiles', addr_words='apricot sun way zebulon', numbers='366', state='nc')
    S2-319331305: 'ROYAL INC CENTER' | '365 APRICOT SUN WAY, ZEBULON, NC'  (name_core='royal center', addr_words='apricot sun way zebulon', numbers='365', state='nc')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-840994883: 'Red Exports Private Limited' | 'No. 10 & No. 13, Srinivasan Street Anna Nagar, Tollgate, Little Kanchipuram, Kanchipuram, Kancheepuram, Tamil Nadu'  (name_core='red exports private', addr_words='and srinivasan anna nagar tollgate little kanchipuram kancheepuram', numbers='10,13', state='tn')
    S2-702780349: 'private red exports limited' | 'NO. 30 & NO. 13, N/A, KANCHIPURAM, KANCHEEPURAM, Tamil Nadu'  (name_core='private red exports', addr_words='and n a kanchipuram kancheepuram', numbers='13,30', state='tn')
  [no shared number]
    S1-494419292: 'Education Coalition' | '15 Kings Road, Norwood, MA'  (name_core='education coalition', addr_words='kings norwood', numbers='15', state='ma')
    S3-661057642: 'Education Coleitein' | 'Kings Rd, Norwood, Massachusetts'  (name_core='education coleitein', addr_words='kings norwood', numbers='', state='ma')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-351563602: 'Ps Plus LLP' | 'Maharashtra, Unit 12 Surya Enclave Chstulsidham, Thane, Thane West'  (name_core='ps plus llp', addr_words='unit surya enclave chstulsidham thane west', numbers='12', state='mh')
    S3-488699561: 'Sri Ps LLP Center' | 'Unit 12- Surya Enclave Chstulsiham, Thane West, Thane, महाराष्ट्र'  (name_core='ps llp center', addr_words='unit surya enclave chstulsiham thane west', numbers='12', state='mh')
  [no shared number]
    S1-85937592: 'Mountain Coalition' | '14931 Pomfret Road, Clackamas, OR'  (name_core='mountain coalition', addr_words='pomfret clackamas', numbers='14931', state='or')
    S2-554215236: 'Mountain Coalition Inc' | 'POMFRET RD, CLACKAMAS, OR'  (name_core='mountain coalition', addr_words='pomfret clackamas', numbers='', state='or')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-98051426: 'Rapid Medical Holdings' | '209 Hassie Lane, Lexington, NC'  (name_core='rapid medical holdings', addr_words='hassie lexington', numbers='209', state='nc')
    S2-708369307: 'Rapid  Medical' | '209 HAHSIE LANE, LEXINGTONT OWNSHIP, NC'  (name_core='rapid medical', addr_words='hahsie lexingtont ownship', numbers='209', state='nc')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-24707733: 'Ss Trading Private Limited' | 'Plot No. 176/P, Survey No.1, Brundavan Colony, Uddamgadda, Mailardevpally, Rajendranaga, R, Hyderabad, Telangana'  (name_core='ss trading private', addr_words='p survey brundavan colony uddamgadda mailardevpally rajendranaga r hyderabad', numbers='1,176', state='tg')
    S2-843884833: 'ఎస్\u200cఎస్ ట్రేడింగ్ ప్రైవేట్ లిమిటెడ్' | 'PLOT NO. 176/P, Telangana, RANGAREDDY, HYDERABAD'  (name_core='estateestate trading private', addr_words='p rangareddy hyderabad', numbers='176', state='tg')
  [no shared number]
    S1-803560234: 'Peak Drugs LLC' | '812 Saturn Circle, Temple, TX'  (name_core='peak drugs', addr_words='saturn temple', numbers='812', state='tx')
    S2-749822111: 'Peak Dgs LLC' | 'SATURN CIR, TEMPLE, TX'  (name_core='peak dgs', addr_words='saturn temple', numbers='', state='tx')