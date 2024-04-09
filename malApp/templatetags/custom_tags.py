import random
from django import template
import random
from malApp.models import *


from malApp.models import Doctor, Register

register = template.Library()

@register.filter(name="check_user_role")
def check_user_role(user):
    try:
        Doctor.objects.get(user=user)
        return "Doctor"
    except:
        try:
            Register.objects.get(user=user)
            return "User"
        except:
            return "Admin"

@register.filter(name="find_mal_score")
def find_mal_score(prob):
    # Generate a random number between 20 and 50
    if 'unhealthy' in prob:
        random_number = random.randint(51, 100)

        # Print the random number
        print(random_number)
        return random_number
    else:
        random_number = random.randint(20, 50)

        # Print the random number
        print(random_number)
        return random_number

@register.filter(name="find_body_mass_index")
def find_body_mass_index(pid):
    data = History.objects.get(id=pid)
    cal = float(data.weight) / float(data.height)**2
    return round(cal, 4)
