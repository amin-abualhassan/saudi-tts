import asyncio,time,json,statistics
from pathlib import Path
from openai import AsyncOpenAI
from app.config import settings
async def main():
 c=AsyncOpenAI(api_key=settings.openai_api_key,max_retries=0);rows=[]
 for text in ['What products does Aljawab offer?','وش تقدمون في الجواب؟','I am very upset.','لا، أقصد البصمة الصوتية.']:
  t=time.monotonic()
  try:
   r=await c.decisions.create(model='gpt-6-luna',input=text,questions=[{'type':'choice','name':'acknowledgment','instructions':'Use a neutral acknowledgment sparingly. none for complaints, corrections, distress, uncertain facts, or requests to stop. Match language.','choices':[{'value':v,'description':v} for v in ['none','ack_ar_tamam','ack_ar_understood','ack_en_aha','ack_en_understood']]}])
   rows.append({'input':text,'elapsed_ms':round((time.monotonic()-t)*1000,1),'answer':r.answers[0].model_dump()})
  except Exception as e:rows.append({'elapsed_ms':round((time.monotonic()-t)*1000,1),'error':type(e).__name__})
 print(json.dumps(rows,ensure_ascii=False));Path('tests/results/decisions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));await c.close()
asyncio.run(main())
