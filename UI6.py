import json
import re
from typing import List

import ollama
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict, ValidationError
