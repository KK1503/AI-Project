const form = document.getElementById('form');
const input = document.getElementById('input');
const chat = document.getElementById('chat');
const sendButton = document.getElementById('send');

function createMessageNode(text, cls='bot'){
  const wrap = document.createElement('div');
  wrap.className = 'message ' + cls;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = cls === 'user' ? 'You' : 'FAQ';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  if(cls === 'bot'){
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
  } else {
    wrap.appendChild(bubble);
    wrap.appendChild(avatar);
  }

  return wrap;
}

function appendMessage(text, cls='bot'){
  const node = createMessageNode(text, cls);
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
}

function setLoading(on){
  if(on){
    sendButton.setAttribute('disabled','');
    input.setAttribute('disabled','');
    const tip = document.createElement('div');
    tip.className = 'message bot typing';
    tip.textContent = 'Thinking...';
    tip.id = '__typing';
    chat.appendChild(tip);
    chat.scrollTop = chat.scrollHeight;
  } else {
    sendButton.removeAttribute('disabled');
    input.removeAttribute('disabled');
    const t = document.getElementById('__typing');
    if(t) t.remove();
  }
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q = input.value.trim();
  if(!q) return;
  appendMessage(q, 'user');
  input.value = '';
  setLoading(true);
  try{
    const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ question: q }) });
    if(!res.ok) throw new Error('Network error');
    const data = await res.json();
    const text = data.answer || data.text || 'No answer found.';
    appendMessage(text, 'bot');
  }catch(err){
    appendMessage('Error: ' + err.message, 'bot');
  }finally{
    setLoading(false);
  }
});

// Allow Enter to send, Shift+Enter for newline in future (if converted to textarea)
input.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    form.requestSubmit();
  }
});
