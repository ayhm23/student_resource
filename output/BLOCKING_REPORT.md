=== Part E: recall measurement on 300k train S1 sample vs full train pools ===
Sample S1 entities: 300000
True (S1,match) pairs among sample: 1037463, across 283216 non-singleton S1 entities

Pair recall BEFORE top-K cut (per key type, and union):
  K1: 779401/1037463 (75.13%)
  K2: 392494/1037463 (37.83%)
  K2_nostate: 28/1037463 (0.00%)
  K3_exact: 481703/1037463 (46.43%)
  K3_pfx8: 376882/1037463 (36.33%)
  K4_exact: 9962/1037463 (0.96%)
  K4_state: 18050/1037463 (1.74%)
  K5: 250890/1037463 (24.18%)
  K5_nostate: 14/1037463 (0.00%)
  K6: 390560/1037463 (37.65%)
  UNION (any key): 946569/1037463 (91.24%)

Union recall BEFORE cut, by country:
  US: 580496/622464 (93.26%)
  India: 366073/414999 (88.21%)

Union recall BEFORE cut, by match source:
  S3: 489994/535974 (91.42%)
  S2: 456575/501489 (91.04%)

Union recall AFTER top-K cut, swept over K:
  K=10: 912452/1037463 (87.95%)
  K=20: 932435/1037463 (89.88%)
  K=30: 939166/1037463 (90.53%)
  K=40: 942574/1037463 (90.85%)
  K=60: 945544/1037463 (91.14%)

No swept K reached 97% union recall; largest K=60 achieved only 91.14%. See 'what I'd try next' in the report.

At K=60: non-singleton S1 with ALL true matches found: 220741/283216 (77.94%); with >=1 found: 278534/283216 (98.35%)

Candidates per S1 at K=60: mean=32.9, median=32, p95=60, max=60, total candidate pairs=9835890

30 random MISSED true pairs (never matched by any key), out of 90894 total misses:
  [no shared name token AND no shared number]
    S1-675933180: 'Dynamic International Private Limited' | 'Kuba Khurdh Rajgarh, Mirzapur, Mirzapur, Uttar Pradesh, Mirzapur, C.O Sri Kadam Singh'  (name_core='dynamic international private', addr_words='kuba khurdh rajgarh mirzapur c o sri kadam singh', numbers='', state='up')
    S3-835996472: 'डायनामिक इंटरनेशनल प्राइवेट लिमिटेड' | 'C.o Sri Kadam Singh, Mirzapur, NULL, उत्तर प्रदेश'  (name_core='ddaaynaamaaik inttrneshnl praaivett limaaittedd', addr_words='c o sri kadam singh mirzapur null uttr prdepradesh', numbers='', state='')
  [no shared number]
    S1-381171331: 'First Gulf Live' | '2259 Island Drive, North Topsail Beach, NC'  (name_core='first gulf live', addr_words='island north topsail beach', numbers='2259', state='nc')
    S2-129094896: 'FIRST 6ÚLF LIVE' | 'ISLAND DRIVE, NORTH TOPSAIL BEACH, NC'  (name_core='first 6ulf live', addr_words='island north topsail beach', numbers='', state='nc')
  [no shared number]
    S1-36627397: 'Family Associates Inc' | '1811 Lafayette Place, Unit A6, Columbus, OH'  (name_core='family associates', addr_words='lafayette unit a6 columbus', numbers='6,1811', state='oh')
    S2-338926858: 'FAMILY INC (ASSOCIATES)' | 'LAFAYETTE PL, COLUMBUS, OH'  (name_core='family associates', addr_words='lafayette columbus', numbers='', state='oh')
  [no shared name token AND no shared number]
    S1-146257685: 'Digital Finance Private Limited' | 'Niranjan Avenmue Qadian Road, Batala, Gurdaspur, Punjab'  (name_core='digital finance private', addr_words='niranjan avenmue qadian batala gurdaspur', numbers='', state='pb')
    S3-552088847: 'ਡਿਜੀਟਲ ਫਾਈਨੈਂਸ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ' | 'Niranjan Avenmue Qadian Road, Batala, PB, Gurdaspur'  (name_core='ddijiittl phaaiinains praaiiveett limttidd', addr_words='niranjan avenmue qadian batala gurdaspur', numbers='', state='pb')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-565071385: 'Jai Estate Private Limited' | '57, Mother Teresa Street Jak Nagar, Thirumullaivoyal, Chennai, Tamil Nadu'  (name_core='jai estate private', addr_words='mother teresa jak nagar thirumullaivoyal chennai', numbers='57', state='tn')
    S3-131032578: 'Jai Ebstase Private Limited' | '57, Chennai, TN'  (name_core='jai ebstase private', addr_words='chennai', numbers='57', state='tn')
  [no shared name_core token]
    S1-161445087: 'Digital Engineering LLP' | 'Room No.1, Plot No.421, Ground Floor Parvati Bhawan, Keuta Sahi Chhaka, Old T, Own, Bhubaneswar, Khordha, Orissa'  (name_core='digital engineering llp', addr_words='room ground floor parvati bhawan keuta sahi chhaka old t own bhubaneswar khordha', numbers='1,421', state='od')
    S2-212710087: 'ଡିଜିଟାଲ୍ ଇଞ୍ଜିନିୟରିଂ ଏଲ୍\u200cଏଲ୍\u200cପି' | 'ROOM NO.1, KHORDHA, BHUBANESWAR, ଓଡ଼ିଶା'  (name_core='ddijittaal inyjiniyyrin elelpi', addr_words='room khordha bhubaneswar odd ishaa', numbers='1', state='')
  [no shared name token AND no shared number]
    S1-439700278: 'Empire Valley Zhejiang' | '148 Bagdad Road, Unit 230, Leander, TX'  (name_core='zhejiang', addr_words='bagdad unit leander', numbers='148,230', state='tx')
    S2-330673541: 'Empire Va1ley  Zhejang' | 'BAGDAD RD, LEANDER, TX'  (name_core='empire va1ley zhejang', addr_words='bagdad leander', numbers='', state='tx')
  [no shared number]
    S1-502169689: 'Little Bike Shop LLC' | 'ME, Biddeford, 349 Guinea Road'  (name_core='little bike shop', addr_words='biddeford guinea', numbers='349', state='me')
    S3-435408097: 'Little Bike Shop Llc Trading' | ''  (name_core='little bike shop trading', addr_words='', numbers='', state='')
  [no shared name token AND no shared number]
    S1-602953298: 'Krishna Business Private Limited' | 'C/O Bimal Prasad Gupta Vill : Pakuahat, Malda, West Bengal'  (name_core='krishna business private', addr_words='c o bimal prasad gupta vill pakuahat malda', numbers='', state='wb')
    S2-371268933: 'কৃষ্ণ বিজনেস প্রাইভেট লিমিটেড' | 'C/O BIMAL PRASAD GUPTA VILL : PAKUAHAT, MALDA, West Bengal'  (name_core='krssnn bijnes praaibhett limittedd', addr_words='c o bimal prasad gupta vill pakuahat malda', numbers='', state='wb')
  [no shared number]
    S1-120031434: 'Colonial Fellowship LLC' | '488 Betts Bridge Road, West Pawlet, VT'  (name_core='colonial fellowship', addr_words='betts bridge west pawlet', numbers='488', state='vt')
    S3-34530811: 'Colonial Fellowship Fellowship LLC' | 'Betts Bridge Rd, W Ppawlet, Vermont'  (name_core='colonial fellowship', addr_words='betts bridge w ppawlet', numbers='', state='vt')
  [no shared number]
    S1-923303276: 'Integrated Chile Inc.' | '704 Vine Street, Murfreesboro, TN'  (name_core='integrated chile', addr_words='vine murfreesboro', numbers='704', state='tn')
    S2-13133739: 'Integrated Chi1e Inc.' | 'MURFREESBORO, TN, 705 VINE STREET'  (name_core='integrated chi1e', addr_words='murfreesboro vine', numbers='705', state='tn')
  [no shared name_core token]
    S1-637281127: 'Bharat Power Private Limited' | '1004, Parshwnath Business Park Nr. Auda Garden, Prahladnagar Vejalpur, Ahmedabad, Gujarat'  (name_core='bharat power private', addr_words='parshwnath business park nr auda garden prahladnagar vejalpur ahmedabad', numbers='1004', state='gj')
    S3-648359953: 'ભારત પાવર પ્રાઇવેટ લિમિટેડ' | '1004, Ahmedabad, GJ'  (name_core='bhaart paavr praaivett limittedd', addr_words='ahmedabad', numbers='1004', state='gj')
  [no shared number]
    S1-558165746: 'Gilemette Cooper, DDS, PC' | '4317 Washington Street, Unit APT 32, Indianapolis, IN'  (name_core='gilemette cooper dds', addr_words='unit apt indianapolis in', numbers='32,4317', state='wa')
    S2-401733856: 'Gi1emette Cooper, DDS, PC' | '317 Washington St, INDIANAPOLIS, IN'  (name_core='gi1emette cooper dds', addr_words='indianapolis in', numbers='317', state='wa')
  [no shared number]
    S1-4637566: 'Software Solo Plastic Malegaon' | 'At Post Chaugaon, Tal Baglan, Satana, Malegaon, Nashik, Maharashtra'  (name_core='software solo plastic malegaon', addr_words='at post chaugaon tal baglan satana malegaon nashik', numbers='', state='mh')
    S2-825980308: 'Software Som Plastic (Malegaon)' | 'AT POST CHAUGAON, TAL BAGLAN, SATANA, MALEGAON, महाराष्ट्र'  (name_core='software som plastic malegaon', addr_words='at post chaugaon tal baglan satana malegaon mhaaraassttr', numbers='', state='')
  [no shared name_core token]
    S1-812620515: 'Unique Developers Private Limited' | '24/1583, Edavalath Thazham Thondayad, Chevarambalam P.O, Kozhikode, Kerala'  (name_core='unique developers private', addr_words='edavalath thazham thondayad chevarambalam p o kozhikode', numbers='24,1583', state='kl')
    S2-691165946: 'യുണീക്ക് ഡെവലപ്പേഴ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്' | '24/1583, KOZHIKODE, Kerala'  (name_core='yunniikk ddevlppeellls praivrrrr limirrrrdd', addr_words='kozhikode', numbers='24,1583', state='kl')
  [no shared number]
    S1-699256542: 'Heart Alliance LLC' | '10161 Liberty Road, Liberty, NC'  (name_core='heart alliance', addr_words='liberty', numbers='10161', state='nc')
    S2-516959746: 'Heart Álliance' | '10195 LIBERTY RD, NC, LIBERTY'  (name_core='heart alliance', addr_words='liberty', numbers='10195', state='nc')
  [no shared number]
    S1-301318789: 'National Association LLC' | '2822 79, Colesville, NY'  (name_core='national association', addr_words='colesville', numbers='79,2822', state='ny')
    S2-383790236: 'NATIONAL LLC ASSOCIATION' | ''  (name_core='national association', addr_words='', numbers='', state='')
  [no shared name_core token]
    S1-809447271: 'Gujarat Care Private Limited' | 'Sr No. 87/1/1, Fl No 203, Nandani Takale Nagar, Haveli, Pune, Maharashtra'  (name_core='gujarat care private', addr_words='sr fl nandani takale nagar haveli pune', numbers='1,87,203', state='mh')
    S2-47809782: 'गुजरात केयर प्राइवेट लिमिटेड' | 'DOOR NO 304 SR NO. 87/1/1, PUNE REGION, PUNE, Maharashtra'  (name_core='gujraat keyr praaivett limaaittedd', addr_words='sr pune region', numbers='1,87,304', state='mh')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-909034211: 'Laxmi Services Private Limited' | '416, Platinum Plaza, Nr. Shakti Enclave, Ahmadabad City, Ahmedabad, Gujarat'  (name_core='laxmi services private', addr_words='platinum plaza nr shakti enclave ahmadabad ahmedabad', numbers='416', state='gj')
    S3-881323095: 'LAXMI 5ERVICES PRIVATE LIMITED' | 'ગુજરાત, Ahmadabad City, 416, Ahmedabad'  (name_core='laxmi 5ervices private', addr_words='gujraat ahmadabad ahmedabad', numbers='416', state='')
  [no shared number]
    S1-280896920: 'Family Precision Clinic Inc' | 'NC, Apex, 400 Homestead Park Drive'  (name_core='family precision clinic', addr_words='apex homestead park', numbers='400', state='nc')
    S2-561859228: 'Family Percision Clinic Inc' | 'APEX, NC, 399 HOMESTEAD PARK DRIVE'  (name_core='family percision clinic', addr_words='apex homestead park', numbers='399', state='nc')
  [no shared number]
    S1-186535878: 'Madalena Smith Dynamic Green' | '302 Heritage Circle, Burnsville, MN'  (name_core='madalena smith dynamic green', addr_words='heritage burnsville', numbers='302', state='mn')
    S2-251850089: 'Madalena Smith Dynamic' | ''  (name_core='madalena smith dynamic', addr_words='', numbers='', state='')
  [no shared name_core token]
    S1-652315195: 'Global Services Private Limited' | '206-207, Shangrila Arcade, Opp. Shyamal Row Houses, Shyamal Cross Road, Satellite, Ahmedabad, Gujarat'  (name_core='global services private', addr_words='shangrila arcade opp shyamal row houses cross satellite ahmedabad', numbers='206,207', state='gj')
    S2-622103286: 'ગ્લોબલ સર્વિસીસ પ્રાઇવેટ લિમિટેડ' | 'A-206-207, AHMEDABAD, Gujarat'  (name_core='globl srvisiis praaivett limittedd', addr_words='a ahmedabad', numbers='206,207', state='gj')
  [no shared name_core token]
    S1-150200129: 'Lakshmi Bright Agro Private Limited' | 'A - 9, Vikramaditya Society, Thakkar Bapa Nagar, N. H. No. 8, Ahmedabad, Gujarat'  (name_core='agro private', addr_words='a vikramaditya society thakkar bapa nagar n h ahmedabad', numbers='8,9', state='gj')
    S2-979746338: 'લક્ષ્મી બ્રાઇટ એગ્રો પ્રાઇવેટ લિમિટેડ' | 'A - 9, AHMEDABAD, Gujarat'  (name_core='lkssmii braaitt egro praaivett limittedd', addr_words='a ahmedabad', numbers='9', state='gj')
  [no shared number]
    S1-447738657: 'Unique Travel Corporation' | 'Firozpur, Bareka, Fazilka, Nihal Khera, Punjab, C/O Naresh Kumar, Fazilka'  (name_core='unique travel', addr_words='firozpur bareka fazilka nihal khera c o naresh kumar', numbers='', state='pb')
    S3-770335928: 'Center Unique Corporation' | 'C/o Naresh Kumar, Bareka, Fazilka, Nihal Khera, Fazilka, Firozpur, PB'  (name_core='center unique', addr_words='c o naresh kumar bareka fazilka nihal khera firozpur', numbers='', state='pb')
  [no shared number]
    S1-303136764: 'Arts Society' | '2707 Toledo Avenue, Alton, IL'  (name_core='arts society', addr_words='toledo alton', numbers='2707', state='il')
    S3-596388765: 'Arts Sóciety Sóciety' | '707 Toledo Avenue, <NULL>, Alton, Illinois'  (name_core='arts society', addr_words='toledo null alton', numbers='707', state='il')
  [no shared name_core token]
    S1-878628285: 'SR Solutions Private Limited' | 'Plot No. C-26C, Jeewan Park Pankha Road, Uttam Nagar, New Delhi, Delhi'  (name_core='sr solutions private', addr_words='c 26c jeewan park pankha uttam nagar', numbers='26', state='dl')
    S3-930743281: 'Solbrix' | 'Pot No. C-26c, Uttam Nagar, New Delhi, DL'  (name_core='solbrix', addr_words='pot c 26c uttam nagar', numbers='26', state='dl')
  [no shared number]
    S1-902129787: 'Heartland Law Group, LLC' | '2304 Colonial Drive, Baytown, TX'  (name_core='heartland law group', addr_words='colonial baytown', numbers='2304', state='tx')
    S3-804916966: 'Heartland Group, LLC Center' | 'Colonial Drive, Baytown, Texas'  (name_core='heartland group center', addr_words='colonial baytown', numbers='', state='tx')
  [no shared number]
    S1-61237513: 'VSS Vegetable Pvt. Ltd.' | 'Sharda Shopping Centre, Second Floor, Near Hare Krishna, Pritamnagar, Paldi, Ahmedabad, Gujarat'  (name_core='vss vegetable pvt', addr_words='sharda shopping centre second floor near hare krishna pritamnagar paldi ahmedabad', numbers='', state='gj')
    S3-847106234: 'VSS Pvt. Ltd. Services' | 'Sharda Shopping Centre, Ahmedabad, ગુજરાત'  (name_core='vss pvt services', addr_words='sharda shopping centre ahmedabad gujraat', numbers='', state='')
  [shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)]
    S1-879248909: 'Main Street Grill' | '2 Jackson Street, Unit 2, Haverhill, MA'  (name_core='main street grill', addr_words='jackson unit haverhill', numbers='2', state='ma')
    S3-722795160: 'Main-Street' | '# 2, 2 Jackaon St, Massachusetts, Haverhill'  (name_core='main street', addr_words='jackaon haverhill', numbers='2', state='ma')
  [no shared number]
    S1-862968649: 'Shakti Hospitality Private Limited' | 'H/O- Annapurna Das Lingapada, Soro, Balasore, Orissa'  (name_core='shakti hospitality private', addr_words='h o annapurna das lingapada soro balasore', numbers='', state='od')
    S3-854822196: 'Shakti Private Limited Services' | 'H/o- Annapurna Das Lingapada, Balasore, OD'  (name_core='shakti private services', addr_words='h o annapurna das lingapada balasore', numbers='', state='od')


=== Part F: full test candidate generation ===
Test S1 entities: 1732544; with >=1 candidate: 1724223 (99.52%); with 0 candidates: 8321

Candidates per S1 by country:
  US: n_S1=663106, mean=30.4, median=28, max=60, zero-candidate S1s=3960 (0.60%)
  France: n_S1=259452, mean=28.4, median=25, max=60, zero-candidate S1s=2016 (0.78%)
  India: n_S1=809986, mean=38.5, median=40, max=60, zero-candidate S1s=2345 (0.29%)

(France has no training labels; a much higher zero-candidate or low-candidate rate for France above than US/India would mean the mined dictionaries/keys, which are all trained on US/India pairs, generalize poorly to French names/addresses.)

Wrote C:\Users\archi\OneDrive\Desktop\sem 7\Amazon ML challenge\run1\student_resource\output\candidate_pairs.tsv (1732544 rows).