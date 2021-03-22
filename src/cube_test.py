# test cube function
from src.cube import cube

mycube=cube()
mycube.turn(0)
mycube.turn(3)

cube2=mycube.copy()

print(mycube.check(mycube.state))

pass