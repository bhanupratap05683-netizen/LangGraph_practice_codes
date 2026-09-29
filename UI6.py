import json
import re
from typing import List

import ollama
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict, ValidationError

MODEL_NAME = "qwen2.5-coder:7b"