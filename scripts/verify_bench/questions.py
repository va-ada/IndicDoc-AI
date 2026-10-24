"""
Parallel English / Hindi / Tamil questions for the verification benchmark.

Numeric templates get fresh values for every paper, so values can be changed to simulate a translation or typing
error. Fixed questions (facts, text options) and short-answer questions add the variety of a real paper.
Hindi and Tamil follow the wording style of NEET / state-board bilingual papers.
"""
import random

LANGS = ('en', 'hi', 'ta')

# --- numeric MCQ templates: stem(lang, **v), values(rng), options(v, rng) -> four numbers, unit per language
TEMPLATES = [
    {
        'stem': {'en': 'A car travels {d} km in {t} hours. What is its average speed?',
                 'hi': 'एक कार {t} घंटे में {d} किमी चलती है। उसकी औसत चाल क्या है?',
                 'ta': 'ஒரு கார் {t} மணி நேரத்தில் {d} கி.மீ. பயணிக்கிறது. அதன் சராசரி வேகம் என்ன?'},
        'values': lambda r: (lambda t: {'t': t, 'd': t * r.choice([30, 40, 45, 50, 60, 70])})(r.choice([2, 3, 4, 5])),
        'answer': lambda v: v['d'] // v['t'],
        'unit': {'en': '{x} km/h', 'hi': '{x} किमी/घंटा', 'ta': '{x} கி.மீ./மணி'},
    },
    {
        'stem': {'en': 'A {v} V battery is connected across a resistor of {r} ohm. What current flows through the resistor?',
                 'hi': '{v} V की एक बैटरी को {r} ओम के प्रतिरोधक के सिरों से जोड़ा गया है। प्रतिरोधक में कितनी धारा प्रवाहित होगी?',
                 'ta': '{v} V மின்கலம் ஒன்று {r} ஓம் மின்தடையின் குறுக்கே இணைக்கப்பட்டுள்ளது. மின்தடையின் வழியே பாயும் மின்னோட்டம் எவ்வளவு?'},
        'values': lambda r: (lambda i: {'r': i[1], 'v': i[0] * i[1]})((r.choice([2, 3, 4, 5]), r.choice([4, 6, 8, 12]))),
        'answer': lambda v: v['v'] // v['r'],
        'unit': {'en': '{x} A', 'hi': '{x} A', 'ta': '{x} A'},
    },
    {
        'stem': {'en': 'A shopkeeper sells an article for Rs {sp} after giving a discount of {p}%. What is its marked price?',
                 'hi': 'एक दुकानदार {p}% की छूट देकर एक वस्तु {sp} रुपये में बेचता है। वस्तु का अंकित मूल्य क्या है?',
                 'ta': 'ஒரு கடைக்காரர் {p}% தள்ளுபடி கொடுத்து ஒரு பொருளை {sp} ரூபாய்க்கு விற்கிறார். அதன் குறித்த விலை என்ன?'},
        'values': lambda r: (lambda p, mp: {'p': p, 'mp': mp, 'sp': mp * (100 - p) // 100})(
            r.choice([10, 20, 25]), r.choice([400, 800, 1200, 1600, 2000])),
        'answer': lambda v: v['mp'],
        'unit': {'en': 'Rs {x}', 'hi': '{x} रुपये', 'ta': '{x} ரூபாய்'},
    },
    {
        'stem': {'en': 'An object of mass {m} kg has a volume of {vol} cubic metres. What is its density?',
                 'hi': '{m} किग्रा द्रव्यमान वाली एक वस्तु का आयतन {vol} घन मीटर है। उसका घनत्व क्या है?',
                 'ta': '{m} கி.கி. நிறையுள்ள ஒரு பொருளின் கனஅளவு {vol} கன மீட்டர். அதன் அடர்த்தி என்ன?'},
        'values': lambda r: (lambda vol, d: {'vol': vol, 'm': vol * d})(r.choice([2, 4, 5]), r.choice([300, 500, 800, 900])),
        'answer': lambda v: v['m'] // v['vol'],
        'unit': {'en': '{x} kg per cubic metre', 'hi': '{x} किग्रा प्रति घन मीटर', 'ta': '{x} கி.கி. / கன மீட்டர்'},
    },
    {
        'stem': {'en': 'A train {l} m long passes a pole in {t} seconds. What is the speed of the train?',
                 'hi': '{l} मीटर लंबी एक रेलगाड़ी एक खंभे को {t} सेकंड में पार करती है। रेलगाड़ी की चाल क्या है?',
                 'ta': '{l} மீட்டர் நீளமுள்ள ஒரு ரயில் ஒரு கம்பத்தை {t} வினாடிகளில் கடக்கிறது. ரயிலின் வேகம் என்ன?'},
        'values': lambda r: (lambda t, s: {'t': t, 'l': t * s})(r.choice([9, 12, 15, 18]), r.choice([10, 15, 20, 25])),
        'answer': lambda v: v['l'] // v['t'],
        'unit': {'en': '{x} m/s', 'hi': '{x} मी/से', 'ta': '{x} மீ/வி'},
    },
    {
        'stem': {'en': 'The average of {n} numbers is {a}. If the number {x} is removed, what is the average of the rest?',
                 'hi': '{n} संख्याओं का औसत {a} है। यदि संख्या {x} हटा दी जाए, तो शेष संख्याओं का औसत क्या होगा?',
                 'ta': '{n} எண்களின் சராசரி {a}. அவற்றிலிருந்து {x} என்ற எண் நீக்கப்பட்டால், மீதமுள்ள எண்களின் சராசரி என்ன?'},
        'values': lambda r: (lambda n, b, k: {'n': n, 'b': b, 'x': b + k * (n - 1), 'a': b + k})(
            r.choice([5, 6, 8, 10]), r.choice([20, 30, 40, 50]), r.choice([2, 3, 4])),
        'answer': lambda v: v['b'],
        'unit': {'en': '{x}', 'hi': '{x}', 'ta': '{x}'},
    },
]

# --- fixed MCQs with numeric options
FIXED_NUMERIC = [
    ({'en': 'In which year was the Constitution of India adopted?',
      'hi': 'भारत का संविधान किस वर्ष अंगीकृत किया गया था?',
      'ta': 'இந்திய அரசியலமைப்பு எந்த ஆண்டு ஏற்றுக்கொள்ளப்பட்டது?'}, [1947, 1948, 1949, 1950], '{x}'),
    ({'en': 'How many bones are there in the body of an adult human?',
      'hi': 'एक वयस्क मनुष्य के शरीर में कितनी हड्डियाँ होती हैं?',
      'ta': 'வயது வந்த மனிதரின் உடலில் எத்தனை எலும்புகள் உள்ளன?'}, [196, 206, 216, 226], '{x}'),
    ({'en': 'At what temperature does pure water boil at sea level?',
      'hi': 'समुद्र तल पर शुद्ध जल किस तापमान पर उबलता है?',
      'ta': 'கடல் மட்டத்தில் தூய நீர் எந்த வெப்பநிலையில் கொதிக்கிறது?'}, [90, 100, 110, 120], '{x} °C'),
    ({'en': 'How many days are there in a leap year?',
      'hi': 'एक अधिवर्ष में कितने दिन होते हैं?',
      'ta': 'ஒரு லீப் ஆண்டில் எத்தனை நாட்கள் உள்ளன?'}, [364, 365, 366, 367], '{x}'),
    ({'en': 'In which year did India become independent?',
      'hi': 'भारत किस वर्ष स्वतंत्र हुआ?',
      'ta': 'இந்தியா எந்த ஆண்டு சுதந்திரம் பெற்றது?'}, [1945, 1946, 1947, 1948], '{x}'),
]

# --- fixed MCQs with text options
FIXED_TEXT = [
    ({'en': 'Which gas do plants absorb during photosynthesis?',
      'hi': 'प्रकाश संश्लेषण के दौरान पौधे कौन-सी गैस अवशोषित करते हैं?',
      'ta': 'ஒளிச்சேர்க்கையின் போது தாவரங்கள் எந்த வாயுவை உறிஞ்சுகின்றன?'},
     {'en': ['Carbon dioxide', 'Oxygen', 'Nitrogen', 'Hydrogen'],
      'hi': ['कार्बन डाइऑक्साइड', 'ऑक्सीजन', 'नाइट्रोजन', 'हाइड्रोजन'],
      'ta': ['கார்பன் டை ஆக்சைடு', 'ஆக்ஸிஜன்', 'நைட்ரஜன்', 'ஹைட்ரஜன்']}),
    ({'en': 'Which planet is known as the Red Planet?',
      'hi': 'किस ग्रह को लाल ग्रह कहा जाता है?',
      'ta': 'எந்தக் கோள் சிவப்புக் கோள் என்று அழைக்கப்படுகிறது?'},
     {'en': ['Mars', 'Venus', 'Jupiter', 'Saturn'], 'hi': ['मंगल', 'शुक्र', 'बृहस्पति', 'शनि'],
      'ta': ['செவ்வாய்', 'வெள்ளி', 'வியாழன்', 'சனி']}),
    ({'en': 'Who wrote the national anthem of India?',
      'hi': 'भारत का राष्ट्रगान किसने लिखा?',
      'ta': 'இந்தியாவின் தேசிய கீதத்தை எழுதியவர் யார்?'},
     {'en': ['Rabindranath Tagore', 'Bankim Chandra Chatterjee', 'Sarojini Naidu', 'Mahatma Gandhi'],
      'hi': ['रवींद्रनाथ टैगोर', 'बंकिम चंद्र चट्टोपाध्याय', 'सरोजिनी नायडू', 'महात्मा गांधी'],
      'ta': ['ரவீந்திரநாத் தாகூர்', 'பங்கிம் சந்திர சட்டர்ஜி', 'சரோஜினி நாயுடு', 'மகாத்மா காந்தி']}),
    ({'en': 'Which is the largest organ of the human body?',
      'hi': 'मानव शरीर का सबसे बड़ा अंग कौन-सा है?',
      'ta': 'மனித உடலின் மிகப்பெரிய உறுப்பு எது?'},
     {'en': ['Skin', 'Liver', 'Heart', 'Lungs'], 'hi': ['त्वचा', 'यकृत', 'हृदय', 'फेफड़े'],
      'ta': ['தோல்', 'கல்லீரல்', 'இதயம்', 'நுரையீரல்']}),
]

# --- short-answer questions (Section B), marks printed with each
SHORT = [
    {'en': 'Explain the process of photosynthesis.', 'hi': 'प्रकाश संश्लेषण की प्रक्रिया समझाइए।',
     'ta': 'ஒளிச்சேர்க்கை செயல்முறையை விளக்குக.'},
    {'en': "State Ohm's law and write its formula.", 'hi': 'ओम का नियम लिखिए और उसका सूत्र दीजिए।',
     'ta': 'ஓம் விதியைக் கூறி அதன் சூத்திரத்தை எழுதுக.'},
    {'en': 'Describe two causes of soil erosion.', 'hi': 'मृदा अपरदन के दो कारणों का वर्णन कीजिए।',
     'ta': 'மண் அரிப்பின் இரண்டு காரணங்களை விவரிக்கவும்.'},
    {'en': 'Write three features of the Indian Constitution.', 'hi': 'भारतीय संविधान की तीन विशेषताएँ लिखिए।',
     'ta': 'இந்திய அரசியலமைப்பின் மூன்று சிறப்பம்சங்களை எழுதுக.'},
    {'en': 'Why does the sky appear blue? Explain briefly.', 'hi': 'आकाश नीला क्यों दिखाई देता है? संक्षेप में समझाइए।',
     'ta': 'வானம் ஏன் நீல நிறமாகத் தெரிகிறது? சுருக்கமாக விளக்குக.'},
    {'en': 'What is the difference between weather and climate?', 'hi': 'मौसम और जलवायु में क्या अंतर है?',
     'ta': 'வானிலைக்கும் காலநிலைக்கும் உள்ள வேறுபாடு என்ன?'},
]

HEADER = {
    'title': {'en': 'Model Question Paper: Science and General Studies',
              'hi': 'आदर्श प्रश्न पत्र: विज्ञान एवं सामान्य अध्ययन',
              'ta': 'மாதிரி வினாத்தாள்: அறிவியல் மற்றும் பொது அறிவு'},
    'meta': {'en': 'Time: 3 Hours        Maximum Marks: {total}',
             'hi': 'समय: 3 घंटे        पूर्णांक: {total}',
             'ta': 'நேரம்: 3 மணி        மொத்த மதிப்பெண்கள்: {total}'},
    'instructions': {'en': 'Instructions: All questions are compulsory. In Section A each question carries 1 mark.',
                     'hi': 'निर्देश: सभी प्रश्न अनिवार्य हैं। खंड क में प्रत्येक प्रश्न 1 अंक का है।',
                     'ta': 'அறிவுரைகள்: அனைத்து வினாக்களுக்கும் விடையளிக்கவும். பகுதி அ-வில் ஒவ்வொரு வினாவுக்கும் 1 மதிப்பெண்.'},
    'section_a': {'en': 'Section A', 'hi': 'खंड क', 'ta': 'பகுதி அ'},
    'section_b': {'en': 'Section B', 'hi': 'खंड ख', 'ta': 'பகுதி ஆ'},
    'marks': {'en': '[{m} marks]', 'hi': '[{m} अंक]', 'ta': '[{m} மதிப்பெண்கள்]'},
}

MARKERS = {'digit': {lang: ['(1)', '(2)', '(3)', '(4)'] for lang in LANGS},
           'letter': {'en': ['(a)', '(b)', '(c)', '(d)'], 'hi': ['(क)', '(ख)', '(ग)', '(घ)'],
                      'ta': ['(அ)', '(ஆ)', '(இ)', '(ஈ)']}}


def numeric_options(answer: int, rng: random.Random):
    """Four distinct positive options around the answer, the answer in a random position."""
    step = max(1, answer // 5) if answer > 10 else 1
    pool = sorted({answer + k * step for k in range(-3, 4)} - {answer})
    pool = [p for p in pool if p > 0]
    opts = rng.sample(pool, 3) + [answer]
    rng.shuffle(opts)
    return opts
