# This folder is for any documents, reports, or images, and anything that doesnt have to do with the algorithms.
---
## How to view Nsys or Sqlite reports
- Open Nsight Systems GUI. You can do it 3 ways:  
  - use your system's search bar, open the nsys-ui.exe on where you installed Nsight Systems
  - open command prompt/terminal/powershell
  - type 'nsys-ui' (wihout apostrophes, and its the esiest way. Just need to specify the .exe in PATH first)
- File > Open... and choose the rep or sqlite file  
  OR
- Drag the files to the GUI

# How to analyze
- Open 'CUDA HW' . 'Kernels' and select the function name, for example findMatches if its Rabin-Karp CUDA
- Right click the function name (findMatches in this example) and click 'Show in Events View'. This will show you when it started and how long it ran for, and if it actually used GPU
- While the function chosen (highlighted with blue background), zoom in on the timeline on the right until the block is visible. Image below for example in red arrow where to zoom
![image](https://github.com/user-attachments/assets/1677cd0f-2231-45a7-adc6-38dcc8138985)
- Hover over the block, after a short while of hovering, it will show all kinds of information. In this case what we need was execution starting and end time as well as memory usage (check 'nsys example.png' below)
![nsys example](nsys_example.png)
