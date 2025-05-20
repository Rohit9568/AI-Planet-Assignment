import React, { useState } from 'react';
import './Tooltip.css'; // Import your CSS file for styling

const Tooltip = ({ text, children }) => {
  const [isVisible, setIsVisible] = useState(false);

  const showTooltip = () => {
    setIsVisible(true);
  };

  const hideTooltip = () => {
    setIsVisible(false);
  };

  return (
    <div className="tooltip-container" onMouseEnter={showTooltip} onMouseLeave={hideTooltip}>
      {children}
      {isVisible && <div className="tooltip">{text}</div>}
    </div>
  );
};

export default Tooltip;
