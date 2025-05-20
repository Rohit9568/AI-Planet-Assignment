import React from 'react';
import "./MessageBox.css";
import Avatar from '../Avatar/Avatar';
import chatLoader from '../assets/chat_loader.gif'


function MessageBox(props) {
  return (
    <div className='message-box'>
      {props.isDisplay?<Avatar imgage={props.image}/>:
      <img src={chatLoader} style={{height:"50px", width:"50px"}} alt="ChatLoader" />
      }
      <span className="text-cont">{props.text}</span>
    </div>
  )
}

export default MessageBox
