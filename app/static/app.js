const form = document.getElementById('form');
const input = document.getElementById('input');
const chat = document.getElementById('chat');

function appendMessage(text, cls='bot'){
  const wrap = document.createElement('div');
  wrap.className = 'message ' + cls;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q = input.value.trim();
  if(!q) return;
  appendMessage(q, 'user');
  input.value = '';
  try{
    const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ question: q }) });
    if(!res.ok) throw new Error('Network error');
    const data = await res.json();
    appendMessage(data.answer || 'No answer found.', 'bot');
  }catch(err){
    appendMessage('Error: ' + err.message, 'bot');
  }
});
