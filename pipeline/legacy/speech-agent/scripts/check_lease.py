"""Verify silent admission expires even while its browser sends heartbeats."""
import asyncio,json,time
from pathlib import Path
import httpx
async def main():
 async with httpx.AsyncClient(trust_env=False,timeout=30) as c:
  first=await c.post('http://[::1]:8002/api/session',json={'language':'ar'});first.raise_for_status()
  headers={'X-Session-Token':first.json()['session_token']}
  start=time.monotonic();attempts=[]
  for _ in range(14):
   await asyncio.sleep(5)
   r=await c.post('http://[::1]:8002/api/heartbeat',headers=headers)
   attempts.append({'elapsed_s':round(time.monotonic()-start,1),'status':r.status_code})
   print(attempts[-1],flush=True)
   if r.status_code==401:
    break
  result={'passed':attempts[-1]['status']==401 and attempts[-1]['elapsed_s']>=59,'attempts':attempts}
  await c.post('http://[::1]:8002/api/end',headers=headers)
  Path('tests/results/lease.json').write_text(json.dumps(result,indent=2))
  assert result['passed']
if __name__=='__main__':asyncio.run(main())
