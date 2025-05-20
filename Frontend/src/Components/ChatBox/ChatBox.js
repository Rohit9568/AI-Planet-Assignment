import './ChatBox.css';
import React ,{ useState,useEffect,useRef} from 'react';
import send from "../assets/send.svg"
import chat_avatar from "../assets/chat_avatar.svg"
import MessageBox from '../MessageBox/MessageBox';
import axios from 'axios';
import SpeechRecognition, { useSpeechRecognition  } from 'react-speech-recognition';
import Tooltip from '../Tooltip/Tooltip';



function ChatBox(props) {
  const audiobutton=useRef(null);
  const [isAudioText,setAudioText] = useState(false);

  const [message,setMessage] = useState("");
  const [chatMessage,setChatMessage] = useState([]);
  const [onGoingChat,setOnGoingChat] = useState(false);

  
  
  
  const {transcript, browserSupportsSpeechRecognition ,resetTranscript} = useSpeechRecognition();
  const startListening = () => SpeechRecognition.startListening({ continuous: true });
  



  useEffect(()=>{
    setMessage(transcript);
  },[transcript])

  useEffect(() => {
    document.querySelector('.message-display').scrollTop = document.querySelector('.message-display').scrollHeight
  }, [chatMessage])

  
  const handleChat = async(e) => {
    e.preventDefault();

    if(message !== "") {
      resetTranscript();

      if(isAudioText === true) {
        handleAudioToText(e);
      }
      
      let tempmessage = message;
      setMessage("");
      setOnGoingChat(true);
      setChatMessage([...chatMessage, {text: tempmessage, isDisplay: true}, {text: "", isDisplay: false}]);
      
      try {
        const response = await axios.post('http://127.0.0.1:8000/chat', {"question": tempmessage});
        const answer = response.data;
        
        if (Array.isArray(answer) && answer.length >= 2) {
          setChatMessage([...chatMessage, 
            {text: answer[answer.length - 2]["content"], isDisplay: true},
            {text: answer[answer.length - 1]["content"], isDisplay: true}
          ]);
        } else {
          throw new Error("Invalid response format");
        }
      } catch (error) {
        console.error("Chat error:", error);
        setChatMessage([...chatMessage, 
          {text: tempmessage, isDisplay: true},
          {text: "Sorry, I couldn't process your question. Please try again.", isDisplay: true}
        ]);
      } finally {
        setOnGoingChat(false);
      }
    }
  }

  const handleChatEnter=(e)=>{
   
    if (e.key === 'Enter') {
      e.preventDefault();
      handleChat(e);
      
    }
  }

  const handleAudioToText=async(e)=>{
    e.preventDefault();
    if(!(props.isUploaded || onGoingChat)){
      if(isAudioText===false){
        setAudioText(true)
        audiobutton.current.style.color ="#0FA958";
        startListening();
      }else{
        setAudioText(false)
        audiobutton.current.style.color ="#4c4e4f";
        SpeechRecognition.stopListening();
        
      }
  }
  }

  const handleChangeMessage =(e)=>{
    setMessage(e.target.value)
  }
  if(!browserSupportsSpeechRecognition){
    return null;
  }
  return (
    <>
    <div className="chat-box">
        <div className="message-display">
          {!props.isUploaded?chatMessage.map((element, index)=>{
           if(index%2 ===0){ 
              return <MessageBox key={index} text={element.text} isDisplay={element.isDisplay}/>
           }else{
              return <MessageBox  key={index} text={element.text} image={chat_avatar} isDisplay={element.isDisplay}/>
           }
          }):
          <div className='upload-text'>Upload PDF to Start</div>
          }
         </div>
        <form className="input">
            <input type="text" className='input-section' placeholder='Send a message...' onChange={handleChangeMessage} value={message} onKeyDown={handleChatEnter} readOnly={props.isUploaded || onGoingChat} />
            
              <button className='text-to-audio' ref={audiobutton} onClick={handleAudioToText} onKeyDown={handleChatEnter}>
              <Tooltip text="Click to enable Speech-to-Text">
                <i className="fa-solid fa-microphone"></i>
              </Tooltip >
              </button>
            
            <button className="send-button" onClick={handleChat}   >
                <img src={send} alt="send" />
            </button>
        </form>
    </div>
     
      
    </>
  )
}

export default ChatBox
