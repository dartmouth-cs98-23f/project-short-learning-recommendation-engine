from typing import List, Literal, Annotated
from pydantic import BaseModel, field_validator, conlist, confloat
import json

# Valid general topic names
valid_general_topics = [
    "Algorithms and Data Structures",
    "Artificial Intelligence and Machine Learning",
    "Computer Architecture",
    "Data Science and Analytics",
    "Database Systems and Management",
    "Human-Computer Interaction",
    "Programming Languages and Software Development",
    "Software Engineering and System Design",
    "Web Development and Internet Technologies",
    "Computer Graphics and Visualization",
    "Theoretical Computer Science",
    "Quantum Computing"
]

class GeneralTopic(BaseModel):
    name: str
    ## complexity is between 0 and 1
    complexity: Annotated[float, confloat(ge=0, le=1)]

    # Validate the name to be one of the valid general topics
    @field_validator('name')
    def name_must_be_valid(cls, v):
        if v not in valid_general_topics:
            raise ValueError(f"{v} is not a valid general topic name")
        return v

class Section(BaseModel):
    title: str
    content: List[str]
    topics: List[str]

class VideoContent(BaseModel):
    introduction: str
    sections: List[Section]
    topics: List[str]
    generalTopics: List[GeneralTopic]

def validate_inference_output(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        VideoContent(**data)
        return True
    except Exception as e:
        print(f"Error validating inference output: {e}")
        return False