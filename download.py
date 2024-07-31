import gdown

url = "https://drive.google.com/u/0/uc?id=1QuE4mo7VTiD2igzNi6B6-RMplL6P8GpQ&export=download"
output = "data.zip"
gdown.download(url, output)