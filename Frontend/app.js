const API_URL = 'http://127.0.0.1:8000';
let authToken = '';  // Store JWT token here

// Show Login Form
function showLogin() {
  document.getElementById('login-form').style.display = 'block';
  document.getElementById('register-form').style.display = 'none';
}

// Show Register Form
function showRegister() {
  document.getElementById('register-form').style.display = 'block';
  document.getElementById('login-form').style.display = 'none';
}

// Show Chat Interface after Login
function showChat() {
  document.getElementById('chat-interface').style.display = 'block';
  document.getElementById('login-form').style.display = 'none';
  document.getElementById('register-form').style.display = 'none';
}

// Register New User
async function register() {
  const email = document.getElementById('register-email').value;
  const password = document.getElementById('register-password').value;
  
  const response = await fetch(`${API_URL}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: 'default', email, password })
  });

  if (response.ok) {
    alert('Registration successful! Now, you can login.');
    showLogin();
  } else {
    const error = await response.json();
    alert(`Error during registration: ${error.detail}`);
  }
}

// Login User
async function login() {
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;

  const response = await fetch(`${API_URL}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: 'default', email, password })
  });

  if (response.ok) {
    const data = await response.json();
    authToken = data.access_token;  // Store JWT token
    alert('Login successful!, auth token: ' + authToken);
    showChat();
  } else {
    const error = await response.json();
    alert(`Invalid credentials: ${error.detail}`);
  }
}

// Send a Message to the Bot
async function sendMessage() {
  const message = document.getElementById('user-message').value;
  if (message.trim() === '') return;

  // Display user's message
  const chatBox = document.getElementById('chat-box');
  const userMessage = document.createElement('div');
  userMessage.classList.add('message', 'user');
  userMessage.textContent = `You: ${message}`;
  chatBox.appendChild(userMessage);

  // Send message to backend with JWT token in body
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      content: message,
      token: authToken  // Send JWT token in the body of the request
    })
  });

  if (response.ok) {
    const data = await response.json();

    // Display the bot's response in the chat on the right side
    const botMessage = document.createElement('div');
    botMessage.classList.add('message', 'bot');
    botMessage.textContent = `CMCA: ${data.reply}`;
    chatBox.appendChild(botMessage);

    // Only show the image on the left side (image-container)
    if (data.image_url) {
      const imageContainer = document.getElementById('image-container');
      const img = document.createElement('img');
      img.src = `${API_URL}${data.image_url}`;  // Full URL to the image
      img.alt = 'Generated Image';
      imageContainer.appendChild(img);
    }

    // Scroll to the bottom of the chat
    chatBox.scrollTop = chatBox.scrollHeight;
  } else {
    alert('Error while getting response');
  }

  // Clear the input field
  document.getElementById('user-message').value = '';
}

// Logout User
function logout() {
  authToken = '';
  showLogin();
}