# Implementation Find-S
# Step-1  : Initiliaze h with most specific hypothesis in H
#           h will reject every instance

h=['phi','phi','phi','phi','phi','phi']   # 'phi' indicate no value is accepted. 'any' indicates every value is accepted.


# Training Data
Data=[['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
      ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
      ['Sunny','Warm','Normal','Strong','Warm','Same','No'],
      ['Sunny','Warm','High','Strong','Cool','Change','Yes']
     ]

# Step-2 : Iterate over training data and replace constraints in hypothesis with more general constraints


'''
Function: isConsistent(hypothesis,data)
        This function check whether given hypothesis is consitent with given data instance.
        It compare each attribute's value in hypothesis with respective attribute's value in data.
'''
def isConsistent(h,d):
    # Check number of attribute is hypothesis is one less than number of attribute in data.
    # Since one attribute in data is class attribute which is not considered in hypothesis.
   
    if len(h)!=len(d)-1:
        print('Number of attributes are not same in hypothesis.')
        return False
    else:
       
        # variable 'matched keeps number of attributes which are consistent in hypothesis.
       
        matched=0        
       
        # Iterate over each attribute in hypothesis
       
        for i in range(len(h)):
           
            # Check if attribute in hypothesis is equal to repsective attribute's value in data instance or
            # it has 'any' value.
           
            if ( (h[i]==d[i]) | (h[i]=='any') ):
               
                # if condition is satisfied then increase matched
               
                matched=matched+1
       
        # Return true if for all attribute's value in data, hypothesis is consistent.
       
        if matched==len(h):
            return True
        else:
            return False

       
'''
Function: makeConsistent(hypothesis,data)
        This function change hypothesis to make it consistent with given data instance.
'''
def makeConsistent(h,d):
   
    # Iterate over each attribute in hypothesis
   
    for i in range(len(h)):
       
        # if ith attribute in hypothesis reject each value. 'phi' indicates that each value is rejected.
       
        if((h[i] == 'phi')):
           
            # Replace ith value in hypothesis with data instance's ith attribute value.
            h[i]=d[i]
           
           
            # if hypothesis ith value is not 'phi' and it is also not equal to ith value in data instance.
           
        elif(h[i]!=d[i]):
           
            # Replace ith value in hypothesis with 'any'. 'any' accept each value for that attribute.
            h[i]='any'
   
    # Return updated hypothesis
    return h



print('Begin : Hypothesis :',h)
print('==========================================')
# Iterate over each data instance in given training data

for d in Data:
   
    # Consider only positive instance ( instance with 'Yes' class)
   
    if d[len(d)-1]=='Yes':
       
        # Check whether hypothesis is consistent with current data instance
       
        if ( isConsistent(h,d)):
           
            # Print hypothesis
           
            print ("Hypothesis :",d)
        else:
           
            # If hypothesis is not consistent then make it consistent with current data instance
           
            h=makeConsistent(h,d)
   
       
        # Print current data instance and updated hypothesis
        print ('Training data         :',d)
        print ('Updated Hypothesis    :',h)
        print()
        print('--------------------------------')
print('==========================================')
print('End: Hypothesis :',h)
