import { Room, RoomEvent, Track, ConnectionState, createLocalAudioTrack, LocalAudioTrack } from 'livekit-client';
import './style.css';

type Lang='ar'|'en';
let lang:Lang='ar',room:Room|null=null,mic:LocalAudioTrack|null=null,events:WebSocket|null=null;
let heartbeat:number|undefined,active=false,starting=false,muted=false,showTranscript=false;
let sessionToken='';
const copy={
 ar:{brand:'الجواب',tag:'محادثة صوتية',eyebrow:'أهلًا بك في الجواب',title:'خلّنا نتكلم.',intro:'تعرّف على الجواب من خلال محادثة بسيطة، بالعربي أو بالإنجليزي.',start:'ابدأ المحادثة',end:'إنهاء',mute:'كتم الميكروفون',unmute:'تشغيل الميكروفون',transcript:'عرض المحادثة',privacy:'محادثة تجريبية · ما نسجل صوتك، وتقدر تنهي المحادثة في أي وقت',hint:'اسمح باستخدام الميكروفون، وتكلم براحتك. تقدر تقاطعني في أي وقت.',idle:'جاهز لما تكون جاهز',listening:'تفضل، أسمعك',thinking:'لحظة أفكّر',speaking:'أنا معك',connecting:'جاري الاتصال…',reconnecting:'نحاول نرجع الاتصال…',ended:'يعطيك العافية. حياك في أي وقت.',busy:'في محادثة ثانية الآن. جرّب بعد شوي.',denied:'نحتاج إذن الميكروفون. اسمح له من إعدادات المتصفح وجرّب مرة ثانية.',error:'ما قدرنا نكمل الاتصال. جرّب مرة ثانية.',unavailable:'نجهّز المحادثة لك. جرّب بعد شوي.',you:'أنت',assistant:'الجواب',interrupted:'تمت المقاطعة',sources:'المعلومات من موقع الجواب',muted:'الميكروفون مكتوم'},
 en:{brand:'Aljawab',tag:'Voice conversation',eyebrow:'WELCOME TO ALJAWAB',title:'Let’s talk.',intro:'Get to know Aljawab through a simple conversation, in Arabic or English.',start:'Start conversation',end:'End',mute:'Mute microphone',unmute:'Unmute microphone',transcript:'Show conversation',privacy:'Test conversation · We don’t record your microphone. End anytime.',hint:'Allow microphone access and speak naturally. You can interrupt me at any time.',idle:'Ready when you are',listening:'Listening',thinking:'Let me think',speaking:'Speaking',connecting:'Connecting…',reconnecting:'Reconnecting…',ended:'Thanks for stopping by. Come back anytime.',busy:'Someone else is chatting right now. Please try again shortly.',denied:'Please allow microphone access in your browser settings, then try again.',error:'We couldn’t continue the connection. Please try again.',unavailable:'We’re getting things ready. Please try again shortly.',you:'You',assistant:'Aljawab',interrupted:'Interrupted',sources:'Information from Aljawab’s website',muted:'Microphone muted'}
};
const connectionCopy={
 ar:{enableAudio:'تشغيل الصوت',network:'ما قدرنا نوصّل الصوت. جرّب تبدّل بين الواي فاي وبيانات الجوال، أو افتح الرابط في Safari أو Chrome.',rate:'في محاولات اتصال كثيرة. انتظر دقيقة وجرّب مرة ثانية.',browser:'افتح الرابط في Safari أو Chrome واسمح باستخدام الميكروفون.',audio:'المتصفح موقف الصوت. اضغط تشغيل الصوت، أو افتح الرابط في Safari أو Chrome.',microphone:'ما قدرنا نشغّل الميكروفون. تأكد من إذن المتصفح وقفّل التطبيقات اللي تستخدمه.'},
 en:{enableAudio:'Enable audio',network:'We couldn’t establish the audio connection. Try switching between Wi-Fi and mobile data, or open this link in Safari or Chrome.',rate:'Too many connection attempts. Wait one minute, then try again.',browser:'Open this link in Safari or Chrome and allow microphone access.',audio:'Your browser blocked audio. Tap Enable audio, or open this link in Safari or Chrome.',microphone:'We couldn’t start the microphone. Check browser microphone access and close other apps using it.'}
};
const app=document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML=`<header><a href="https://aljawab.ai" target="_blank" rel="noopener" class="brand"><span class="mark">ا</span><span id="brand"></span><span id="tag"></span></a><button id="language" class="language">English</button></header><main><div id="eyebrow" class="eyebrow"></div><h1 id="title"></h1><p id="intro" class="intro"></p><section class="conversation" aria-label="Voice conversation"><div id="orb" class="orb"><div class="ring"></div><div class="wave"><i></i><i></i><i></i><i></i><i></i></div></div><p id="status" role="status" aria-live="polite"></p><p id="hint" class="hint"></p><div class="controls"><button id="start" class="primary"></button><button id="mute" class="secondary" hidden></button><button id="end" class="end" hidden></button></div><div id="error" class="error" role="alert" hidden></div></section><label class="transcript-toggle"><input type="checkbox" id="transcript-toggle"><span id="transcript-label"></span></label><div id="transcript" class="transcript" aria-live="polite" hidden></div><p id="privacy" class="privacy"></p></main><footer><span id="footer-brand">الجواب / Aljawab</span><a id="sources" href="https://aljawab.ai" target="_blank" rel="noopener"></a><span>العربية & English</span></footer><div id="audio"></div>`;
const el=(id:string)=>document.getElementById(id)!;
const audioButton=document.createElement('button');
audioButton.id='enable-audio';audioButton.className='secondary';audioButton.hidden=true;
el('end').before(audioButton);
function render(){
 document.documentElement.lang=lang;document.documentElement.dir=lang==='ar'?'rtl':'ltr';
 for(const key of ['brand','tag','eyebrow','title','intro','hint','privacy','sources']) el(key).textContent=copy[lang][key as keyof typeof copy.ar];
 el('language').textContent=lang==='ar'?'English':'العربية';
 el('start').textContent=copy[lang].start;el('end').textContent=copy[lang].end;
 el('mute').textContent=muted?copy[lang].unmute:copy[lang].mute;
 el('transcript-label').textContent=copy[lang].transcript;
 el('start').hidden=active; (el('start') as HTMLButtonElement).disabled=starting;
 el('end').hidden=!active;el('mute').hidden=!active;
 audioButton.textContent=connectionCopy[lang].enableAudio;
 audioButton.hidden=!active||!room||room.canPlaybackAudio;
}
function state(value:string){
 el('orb').className='orb '+value;
 const key=value as keyof typeof copy.ar;
 el('status').textContent=muted&&active?copy[lang].muted:(copy[lang][key]||copy[lang].listening);
}
function error(key:'busy'|'denied'|'error'|'unavailable'|'network'|'rate'|'browser'|'audio'|'microphone'){
 el('error').hidden=false;el('error').textContent={...copy[lang],...connectionCopy[lang]}[key];
}
async function api(path:string,body:object={},token=sessionToken,signal?:AbortSignal){
 const result=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json',...(token?{'X-Session-Token':token}:{})},body:JSON.stringify(body),credentials:'same-origin',signal});
 if(!result.ok)throw new Error(String(result.status));
 return result.json();
}
async function reportFailure(stage:string,cause:unknown){
 if(!sessionToken||!['media','publish','ready'].includes(stage))return;
 const platform=/iPhone|iPad|iPod/.test(navigator.userAgent)||(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1)?'ios':/Android/.test(navigator.userAgent)?'android':'other';
 const name=cause instanceof Error?cause.name:'';
 const reason=cause instanceof Error&&'reasonName' in cause&&cause.reasonName==='Timeout'?'timeout':name==='NotAllowedError'?'permission':stage==='media'?'network':'other';
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),1500);
 try{await api('connection-report',{stage,reason,platform},sessionToken,controller.signal);}catch{}finally{clearTimeout(timer);}
}
let provisional:HTMLParagraphElement|null=null;
function transcript(data:{role:string,text:string,final:boolean,interrupted?:boolean}){
 if(!showTranscript)return;
 if(data.role==='user'&&provisional){provisional.remove();provisional=null;}
 const p=document.createElement('p');p.className=data.role+(data.final?'':' provisional');
 const label=document.createElement('strong');label.textContent=(data.role==='user'?copy[lang].you:copy[lang].assistant)+': ';
 p.append(label,document.createTextNode(data.text));
 if(data.interrupted){const note=document.createElement('small');note.textContent=' · '+copy[lang].interrupted;p.append(note);}
 el('transcript').append(p);
 if(data.role==='user'&&!data.final)provisional=p;
 while(el('transcript').children.length>30)el('transcript').firstElementChild?.remove();
 p.scrollIntoView({block:'nearest',behavior:'smooth'});
}
async function stop(send=true){
 const token=sessionToken;sessionToken='';
 active=false;starting=false;clearInterval(heartbeat);events?.close();events=null;
 mic?.stop();mic=null;
 render();state('ended');
 const previous=room;room=null;
 if(previous)await previous.disconnect();
 el('audio').replaceChildren();el('transcript').replaceChildren();provisional=null;muted=false;
 if(send&&token)try{await api('end',{},token);}catch{}
 render();state('ended');
}
async function start(){
 if(starting||active)return;
 starting=true;el('error').hidden=true;render();state('connecting');
 let allocated=false,stage='microphone';
 try{
  if(!navigator.mediaDevices?.getUserMedia||!window.RTCPeerConnection){await stop(false);error('browser');return;}
  room=new Room({adaptiveStream:true,dynacast:true});
  room.on(RoomEvent.AudioPlaybackStatusChanged,()=>render());
  // Attempt unlocking directly in the Start tap. Never hold connection setup
  // on autoplay permission; the visible recovery button handles a refusal.
  void room.startAudio().catch(()=>render());
  mic=await createLocalAudioTrack({echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1});
  stage='session';
  const session=await api('session',{language:lang});allocated=true;
  sessionToken=session.session_token;
  room.on(RoomEvent.TrackSubscribed,(track)=>{
   if(track.kind===Track.Kind.Audio){const audio=track.attach();audio.setAttribute('playsinline','');el('audio').append(audio);}
  });
  room.on(RoomEvent.Reconnecting,()=>state('reconnecting'));
  room.on(RoomEvent.Reconnected,()=>state('listening'));
  room.on(RoomEvent.Disconnected,()=>{if(active){void stop();error('error');}});
  stage='media';
  await room.connect(location.origin.replace('https:','wss:').replace('http:','ws:'),session.token);
  stage='publish';
  await room.localParticipant.publishTrack(mic);
  active=true;starting=false;render();state('listening');
  events=new WebSocket(location.origin.replace('https:','wss:').replace('http:','ws:')+'/api/events',['aljawab','session.'+sessionToken]);
  events.onmessage=event=>{
   const data=JSON.parse(event.data);
   if(data.type==='state')state(data.state);
   else if(data.type==='transcript')transcript(data);
   else if(data.type==='error'){void stop();error('error');}
   else if(data.type==='ended'){void stop(false).then(()=>{if(data.reason==='idle')el('status').textContent=lang==='ar'?'انتهت المحادثة بعد دقيقة بدون كلام. تقدر تبدأ من جديد.':'Conversation closed after one minute without speech. You can start again.';});}
  };
  heartbeat=window.setInterval(()=>void api('heartbeat').catch(()=>{void stop(false);error('error');}),15000);
  stage='ready';await api('ready');
 }catch(e){
  const message=e instanceof Error?e.message:'';
  await reportFailure(stage,e);
  await stop(allocated);
  error(message==='429'?'rate':message==='409'?'busy':message==='503'?'unavailable':e instanceof DOMException&&e.name==='NotAllowedError'?'denied':stage==='microphone'?'microphone':stage==='media'?'network':'error');
 }
}
audioButton.onclick=()=>{if(room)void room.startAudio().then(()=>{el('error').hidden=true;render();}).catch(()=>error('audio'));};
el('start').onclick=()=>void start();
el('end').onclick=()=>void stop();
el('mute').onclick=async()=>{if(!mic)return;muted=!muted;if(muted)await mic.mute();else await mic.unmute();render();state('listening');};
el('language').onclick=()=>{lang=lang==='ar'?'en':'ar';render();state(active?'listening':'idle');if(active)void api('language',{language:lang});};
el('transcript-toggle').onchange=event=>{showTranscript=(event.target as HTMLInputElement).checked;el('transcript').hidden=!showTranscript;if(!showTranscript){el('transcript').replaceChildren();provisional=null;}};
window.addEventListener('pagehide',()=>{mic?.stop();if(active)void fetch('/api/end',{method:'POST',headers:{'X-Session-Token':sessionToken},keepalive:true});});
render();state('idle');
