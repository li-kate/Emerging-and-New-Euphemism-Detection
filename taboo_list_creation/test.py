import pandas as pd
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import re
import requests


BASE_URL = "https://unofficialurbandictionaryapi.com/api/date"

requests.get(
    BASE_URL,
    params={
        "date": "2024-01-01",
        "multiPage": "1,10"
    },
    timeout=30
)