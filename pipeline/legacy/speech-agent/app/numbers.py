"""Spell numbers before speech, including numeric runs split across LLM deltas."""
import re
import unicodedata

EN_ONES = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
EN_TENS = ('zero ten twenty thirty forty fifty sixty seventy eighty ninety').split()
AR_ONES = ('صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة أحدعش اثنعش ثلاثطعش أربعطعش خمسطعش ستطعش سبعطعش ثمانطعش تسعطعش').split()
AR_TENS = ('صفر عشرة عشرين ثلاثين أربعين خمسين ستين سبعين ثمانين تسعين').split()
AR_HUNDREDS = ('صفر مية ميتين ثلاثمية أربعمية خمسمية ستمية سبعمية ثمانمية تسعمية').split()
SCALES = [(10**18,'quintillion',('كوينتليون','كوينتليونين','كوينتليونات')),
          (10**15,'quadrillion',('كوادريليون','كوادريليونين','كوادريليونات')),
          (10**12,'trillion',('تريليون','تريليونين','تريليونات')),
          (10**9,'billion',('مليار','مليارين','مليارات')),
          (10**6,'million',('مليون','مليونين','ملايين')),
          (1000,'thousand',('ألف','ألفين','آلاف'))]
NUMBER = re.compile(r'[+\-−]?(?:\d{1,3}(?:[,٬]\d{3})+|\d+)(?:[.٫]\d+)?(?:\s*[%٪])?')
TAIL = re.compile(r'[+\-−\d,٬.٫%٪]+$')

def integer_words(n,language):
    ar=language=='ar';ones=AR_ONES if ar else EN_ONES
    if n<20:return ones[n]
    if n<100:
        tens,unit=divmod(n,10)
        return (ones[unit]+' و'+AR_TENS[tens] if unit else AR_TENS[tens]) if ar else EN_TENS[tens]+(' '+ones[unit] if unit else '')
    if n<1000:
        hundreds,rest=divmod(n,100)
        head=AR_HUNDREDS[hundreds] if ar else ones[hundreds]+' hundred'
        return head+((' و' if ar else ' and ')+integer_words(rest,language) if rest else '')
    for scale,en,forms in SCALES:
        if n>=scale:
            count,rest=divmod(n,scale)
            if ar:
                head=forms[0] if count==1 else forms[1] if count==2 else integer_words(count,language)+' '+(forms[2] if 3<=count<=10 else forms[0])
            else:head=integer_words(count,language)+' '+en
            return head+((' و' if ar else ' and ' if rest<100 else ' ')+integer_words(rest,language) if rest else '')

def numbers_to_words(text,language='ar'):
    def convert(match):
        raw=match.group();percent=raw.rstrip().endswith(('%','٪'))
        raw=raw.rstrip(' %٪').replace(',','').replace('٬','').replace('٫','.')
        raw=''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in raw)
        sign=''
        if raw[0] in '+-−':
            symbol,raw=raw[0],raw[1:]
            if match.start() and text[match.start()-1].isalnum():sign=symbol
            elif symbol in '-−':sign='سالب ' if language=='ar' else 'minus '
            else:sign='موجب ' if language=='ar' else 'plus '
        whole,_,fraction=raw.partition('.')
        # Avoid huge integer expansion, and preserve leading zeros in identifiers.
        if len(whole)>21 or len(whole)>1 and whole.startswith('0'):
            words=' '.join((AR_ONES if language=='ar' else EN_ONES)[int(c)] for c in whole)
        else:words=integer_words(int(whole),language)
        if fraction:
            words+=(' فاصلة ' if language=='ar' else ' point ')+' '.join((AR_ONES if language=='ar' else EN_ONES)[int(c)] for c in fraction)
        if percent:words+=' بالمية' if language=='ar' else ' percent'
        return sign+words
    return NUMBER.sub(convert,text)

async def spoken_chunks(source,language):
    buffered=''
    async for chunk in source:
        buffered+=chunk
        tail=TAIL.search(buffered)
        cut=tail.start() if tail and any(c.isdecimal() for c in tail.group()) else len(buffered)
        if cut:
            yield numbers_to_words(buffered[:cut],language)
            buffered=buffered[cut:]
    if buffered:yield numbers_to_words(buffered,language)
