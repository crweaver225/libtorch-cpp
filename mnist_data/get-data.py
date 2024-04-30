import requests
import os

def download_file(url, filename):
    """ Helper function to download a file from a URL to the specified filename """
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

def download_mnist_ubyte():
    """ Download the MNIST dataset in UBYTE format """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for file in files:
        url = f"{base_url}{file}"
        download_file(url, file)

# Run the function to download the dataset
download_mnist_ubyte()