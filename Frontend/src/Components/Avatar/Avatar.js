import React from 'react';
import './Avatar.css'

function Avatar({imgage}) {
  return (
    <>
    {imgage ? (<img src={imgage} className='avatar' alt="logo" style={{backgroundColor:"white"}} />)
    :
    (<div className="avatar" data-label="S">

    </div>)
  }
  </>
  

  )
}

export default Avatar
