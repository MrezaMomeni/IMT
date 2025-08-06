📊 Streamlit-Interactive Data Visualization App
This project is a Streamlit-based web application that allows users to upload, visualize, and explore datasets interactively. Built entirely in Python using popular data science libraries like pandas, matplotlib, and seaborn, this tool is perfect for quickly getting insights from your data.

🚀 Features
📁 CSV Upload – Drag-and-drop or browse to upload your dataset.

🧮 Data Summary – Quick look at the structure: column types, missing values, basic stats.

📈 Interactive Visualizations – Choose columns to plot:

Line Plot

Bar Chart

Scatter Plot

Histogram

Heatmap (correlation matrix)

🎛️ Custom Options – Pick your x/y axes, aggregation types, and number of bins.

✨ User-Friendly UI – Powered by Streamlit widgets like selectbox, slider, and checkbox.

🧰 Tech Stack
Python 🐍

Streamlit 🎈

Pandas 🐼

Matplotlib 📉

Seaborn 🐚

📦 Installation
Make sure you have Python 3.8+ installed. Then:


# Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
🏃‍♂️ Running the App
 
streamlit run your_notebook_script.py
Or if it's still a .ipynb, you can:

Convert the notebook to Python:


jupyter nbconvert --to script Untitled17.ipynb
Then run it with Streamlit:


streamlit run Untitled17.py
📁 File Structure

├── Untitled17.ipynb         # Main notebook (app logic)
├── README.md                # This file
├── requirements.txt         # Python dependencies
🧪 Example Dataset
You can test the app with any CSV file, or use a built-in dataset by converting it to CSV (e.g., Iris or Titanic from seaborn).

