import pytest
from app.numbers import numbers_to_words,spoken_chunks

@pytest.mark.parametrize('text,language,expected',[
 ('160','en','one hundred and sixty'),
 ('١٦٠','ar','مية وستين'),
 ('256','ar','ميتين وستة وخمسين'),
 ('١٤٠ مللي ثانية','ar','مية وأربعين مللي ثانية'),
 ('94%','ar','أربعة وتسعين بالمية'),
 ('23','ar','ثلاثة وعشرين'),
 ('1,234','en','one thousand two hundred and thirty four'),
 ('٣٫١٤','ar','ثلاثة فاصلة واحد أربعة'),
 ('-2.5','en','minus two point five'),
 ('2026','ar','ألفين وستة وعشرين'),
 ('007','en','zero zero seven'),
])
def test_spoken_numbers(text,language,expected):
 assert numbers_to_words(text,language)==expected

@pytest.mark.asyncio
async def test_numeric_deltas_are_not_verbalized_as_separate_values():
 async def source():
  for x in ['It has ','2','5','6',' dimensions and ','3.','1','4',' units.']:yield x
 result=''.join([x async for x in spoken_chunks(source(),'en')])
 assert result=='It has two hundred and fifty six dimensions and three point one four units.'

@pytest.mark.asyncio
async def test_final_number_flushes_without_a_trailing_space():
 async def source():
  yield 'العدد '
  yield '١'
  yield '٦٠'
 assert ''.join([x async for x in spoken_chunks(source(),'ar')])=='العدد مية وستين'
