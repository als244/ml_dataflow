# Import the 'app' instance from your main application file (run.py in this example)
from run import app

# This part below is generally NOT executed by Gunicorn,
# but can be useful if you ever try to run wsgi.py directly with Python.
if __name__ == "__main__":
    # Note: Gunicorn doesn't use app.run(). It runs the 'app' object directly.
    # You might put different logic here for direct execution if needed.
    pass
