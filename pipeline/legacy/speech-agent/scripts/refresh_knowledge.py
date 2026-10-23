"""Bounded same-site rendered crawl. Commit a snapshot only after validation."""
import asyncio, datetime, hashlib, json, re
from urllib.parse import urlsplit
from playwright.async_api import async_playwright
from app.config import ROOT

URLS=['https://aljawab.ai/','https://aljawab.ai/#/technologies']
async def refresh():
 pages=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(args=['--no-sandbox'])
  try:
   for url in URLS:
    page=await browser.new_page()
    await page.route('**/*',lambda route: route.continue_() if urlsplit(route.request.url).hostname in {'aljawab.ai','www.aljawab.ai'} else route.abort())
    await page.goto(url,wait_until='networkidle',timeout=45000)
    if urlsplit(page.url).hostname not in {'aljawab.ai','www.aljawab.ai'}:raise RuntimeError('Unapproved redirect')
    # Open FAQ answers; do not submit any forms or follow external links.
    faq=[]
    for button in await page.locator('button').all():
     label=(await button.inner_text()).strip()
     if '?' in label and len(label)<150:
      try:
       await button.click(timeout=500)
       faq.append(await button.evaluate('(e)=>e.parentElement.innerText'))
      except Exception: pass
    chunks=await page.locator('h1,h2,h3,h4,p,li').evaluate_all('(es)=>es.filter(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=="hidden").map(e=>e.innerText)')
    chunks+=faq
    seen=set(); lines=[]
    for chunk in chunks:
     chunk=re.sub(r'\s+',' ',chunk).strip()
     if len(chunk)<12 or chunk in seen:continue
     if re.search(r'ignore (previous|all)|system prompt|developer message',chunk,re.I):continue
     seen.add(chunk);lines.append(chunk)
    text='\n'.join(lines)
    if len(text)<1800 or not all(term in text for term in ['Aljawab','voice']):raise RuntimeError('No usable company content')
    links=await page.locator('a[href^="mailto:"]').evaluate_all('(es)=>es.map(e=>e.href)')
    pages.append(dict(url=url,title=await page.title(),headings=await page.locator('h1,h2,h3').all_inner_texts(),
                      language='ar' if len(re.findall(r'[\u0600-\u06ff]',text))>len(re.findall('[A-Za-z]',text)) else 'en',
                      html_language=await page.locator('html').get_attribute('lang'),text=text,contact_links=sorted(set(links)),
                      retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(text.encode()).hexdigest()))
    await page.close()
  finally: await browser.close()
 target=ROOT/'knowledge/snapshot.json';tmp=target.with_suffix('.tmp')
 tmp.write_text(json.dumps({'pages':pages},ensure_ascii=False,indent=2));tmp.replace(target)
 print('Verified snapshot:',len(pages),'pages,',sum(len(p['text']) for p in pages),'characters')
if __name__=='__main__':asyncio.run(refresh())
