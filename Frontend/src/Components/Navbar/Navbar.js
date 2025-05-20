import './Navbar.css';

import React ,{useRef, useState ,useEffect} from 'react';
import axios from "axios"


import logo from '../assets/logo.svg';
import add from '../assets/add.svg';
import file from '../assets/file_image.svg'


function Navbar(props) {
  const fileInputRef = useRef(null);
  const uploadedFile = useRef(null);

  

  
  
 
  const [uploadedFileName,setUploadedFileName] = useState("");
 

  useEffect(() => {
  },[uploadedFileName]);


  const handleUploadButtonClick = () => {
    fileInputRef.current.click();
  };
  const handleFileChange = async(event) => {
    const selectedFile = event.target.files[0];
    const name = selectedFile["name"];

    const shortenName = name => name.length > 9 ? name.substring(0, 5) + "...pdf" : name;

    setUploadedFileName(shortenName(name));
    const formData = new FormData();
    formData.append('file', selectedFile);

    props.loadingBar.current.continuousStart();
    try {
      const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      props.loadingBar.current.staticStart();
      
      if(response.data["message"] === "Success"){
        props.setIsUploaded(false); // Set to false to enable chat
        uploadedFile.current.style.display = "flex";
      } else {
        throw new Error("Upload failed");
      }
      
    } catch (error) {
      console.error("Upload error:", error);
      setUploadedFileName("Try Again");
      uploadedFile.current.style.color = "red";
      uploadedFile.current.style.display = "flex";
      props.setIsUploaded(true); // Keep chat disabled on error
    }
    props.loadingBar.current.complete();
  };

  return (
    
    <div className="navbar">
      <img src={logo} alt="logo" />
      <div className='right-nav'>
      <div className="uploaded-file" ref={uploadedFile}>
        <img src={file} alt="file"/>
        <div>{uploadedFileName}</div>
      </div>
      <button className="upload-button" onClick={handleUploadButtonClick}>
        <img src={add} alt="add" />
        <div className='button-name'>Upload PDF</div>
        <input type="file" className='hide-input' onChange={handleFileChange} accept='.pdf'  ref={fileInputRef}/>
      </button>
      </div>
      
    </div>
  )
}

export default Navbar
