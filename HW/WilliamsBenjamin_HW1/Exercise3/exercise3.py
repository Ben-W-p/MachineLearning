n = 100
open_lockers = [False] * (n + 1) 

for student in range(1, n + 1):
    #for each student, we start at the students number, and add the number of the student to follow
    #the described pattern
    for locker in range(student, n + 1, student):
        #All students, even the first and second, change the state of every locker they visit
        open_lockers[locker] = not open_lockers[locker]

#Find the open lockers
result = [i for i in range(1, n + 1) if open_lockers[i]]
print(result)
print('Interesting, it ends up being the perfect squares up to 100!')
