
import './App.css';
import Navbar from './Components/Navbar/Navbar';
import ChatBox from './Components/ChatBox/ChatBox';
import LoadingBar from 'react-top-loading-bar';
import { useRef, useState } from 'react';
function App() {
  const loadingBar = useRef(null);

  const [isUploaded,setIsUploaded]=useState(true)
  return (
  <>
  <Navbar loadingBar={loadingBar} setIsUploaded={setIsUploaded}/>
  <LoadingBar color='#0FA958' ref={loadingBar} />
  <ChatBox isUploaded={isUploaded} setIsUploaded={setIsUploaded}/>
  



  </>
  );
}

export default App;
