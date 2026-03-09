from pydantic import BaseModel, EmailStr,AnyUrl, Field, field_validator,model_validator,BeforeValidator, computed_field, model_serializer
from typing import List, Annotated


class Address(BaseModel): 
    house_no: int
    landmark: str
    city: str
    state: str
    pincode: int     


StrClean = Annotated[str, BeforeValidator(lambda v: ' '.join(v.split()))]

class Resume_Validation(BaseModel): 
    name: StrClean
    email: EmailStr
    address: Address
    experience_years: Annotated[int, Field(gt=0, lt=50)]
    skills: List[str]
    cover_letter: Annotated[str, Field(min_length=20)]
    github_url: AnyUrl


    @field_validator('email')
    @classmethod
    def validate_email(cls, value): 
        domains = ['gmail.com', 'outlook.com']

        if value.split('@')[1] not in domains:
            raise ValueError('Enter a valid email')

        return value 

    @model_validator(mode='after')
    def validate_skill(self): 
        updated_list = []

        for skill in self.skills: 
            updated_list.append(skill[0].upper() + skill[1:])

        self.skills = updated_list
        return self

    
    @computed_field
    @property
    def calculate_score(self) -> int:
        total = 0

        if self.experience_years < 10: 
            total += 10
        elif self.experience_years > 20: 
            total += 40
        else: 
            total += 5


        if len(self.skills) > 5: 
            total += 20
        elif len(self.skills) > 3:
            total += 10
        else: 
            total += 5 

        if len(self.cover_letter) > 50 : 
            total += 20
        elif len(self.cover_letter) > 30:
            total += 10
        else: 
            total += 5

        if self.github_url:
            total += 10

        return total

    
    @model_serializer
    def serialize_for_llm(self) -> dict: 
        return {
            'candidate_name': self.name,
            'contact_info': {
                'email': self.email, 
                'github': str(self.github_url)
            }, 
            'location': self.address.model_dump(), 
            'experience_years': self.experience_years, 
            'cover_letter': self.cover_letter,
            'skills': self.skills, 
            'score': self.calculate_score,
            'evaluation': 'strong' if self.calculate_score > 70 else 'average' if self.calculate_score >= 40 else 'weak'
        }


resume_validation = Resume_Validation()

