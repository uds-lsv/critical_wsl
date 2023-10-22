Agnews Topic classification dataset

https://github.com/weakrules/Denoise-multi-weak-sources/blob/master/rules-noisy-labels/Agnews/angews_rule.py


# Labels 
"0": "World",
"1": "Sports",
"2": "Business",
"3": "Sci/Tech"







# Labeling functions (all 9 lf are keyword lf)

## LF1  0: world

r1 = ["atomic", "captives", "baghdad", "israeli", "iraqis", "iranian", "afghanistan", "wounding", "terrorism", "soldiers", \
"palestinians", "palestinian", "policemen", "iraqi", "terrorist", 'north korea', 'korea', \
'israel', 'u.n.', 'egypt', 'iran', 'iraq', 'nato', 'armed', 'peace']


## LF2  0: world

r2= [' war ', 'prime minister', 'president', 'commander', 'minister',  'annan', "military", "militant", "kill", 'operator']




## LF3  1: sports

r3 = ["goals", "bledsoe", "coaches",  "touchdowns", "kansas", "rankings", "no.", \
    "champ", "cricketers", "hockey", "champions", "quarterback", 'club', 'team',  'baseball', 'basketball', 'soccer', 'football', 'boxing',  'swimming', \
    'world cup', 'nba',"olympics","final", "finals", 'fifa',  'racist', 'racism'] 



## LF4   1: sports

r4 = ['athlete',  'striker', 'defender', 'goalkeeper',  'midfielder', 'shooting guard', 'power forward', 'point guard', 'pitcher', 'catcher', 'first base', 'second base', 'third base','shortstop','fielder']




## LF5   1: sports

r5=['lakers','chelsea', 'piston','cavaliers', 'rockets', 'clippers','ronaldo', \
    'celtics', 'hawks','76ers', 'raptors', 'pacers', 'suns', 'warriors','blazers','knicks','timberwolves', 'hornets', 'wizards', 'nuggets', 'mavericks', 'grizzlies', 'spurs', \
    'cowboys', 'redskins', 'falcons', 'panthers', 'eagles', 'saints', 'buccaneers', '49ers', 'cardinals', 'texans', 'seahawks', 'vikings', 'patriots', 'colts', 'jaguars', 'raiders', 'chargers', 'bengals', 'steelers', 'browns', \
    'braves','marlins','mets','phillies','cubs','brewers','cardinals', 'diamondbacks','rockies', 'dodgers', 'padres', 'orioles', 'sox', 'yankees', 'jays', 'sox', 'indians', 'tigers', 'royals', 'twins','astros', 'angels', 'athletics', 'mariners', 'rangers', \
    'arsenal', 'burnley', 'newcastle', 'leicester', 'manchester united', 'everton', 'southampton', 'hotspur','tottenham', 'fulham', 'watford', 'sheffield','crystal palace', 'derby', 'charlton', 'aston villa', 'blackburn', 'west ham', 'birmingham city', 'middlesbrough', \
    'real madrid', 'barcelona', 'villarreal', 'valencia', 'betis', 'espanyol','levante', 'sevilla', 'juventus', 'inter milan', 'ac milan', 'as roma', 'benfica', 'porto', 'getafe', 'bayern', 'schalke', 'bremen', 'lyon', 'paris saint', 'monaco', 'dynamo']



 
## LF6  3: tech

r6 = ["technology", "engineering", "science", "research", "cpu", "windows", "unix", "system", 'computing',  'compute']#, "wireless","chip", "pc", ]




## LF7  3: tech

r7= ["google", "apple", "microsoft", "nasa", "yahoo", "intel", "dell", \
    'huawei',"ibm", "siemens", "nokia", "samsung", 'panasonic', \
    't-mobile', 'nvidia', 'adobe', 'salesforce', 'linkedin', 'silicon', 'wiki'
]




## LF8  - 2:business

r8= ["stock", "account", "financ", "goods", "retail", 'economy', 'chairman', 'bank', 'deposit', 'economic', 'dow jones', 'index', '$',  'percent', 'interest rate', 'growth', 'profit', 'tax', 'loan',  'credit', 'invest']




## LF9  - 2:business

r9= ["delta", "cola", "toyota", "costco", "gucci", 'citibank', 'airlines']