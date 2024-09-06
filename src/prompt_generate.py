def get_prompt(entity_type,
                type_description,
                ):

    prompt = f'''You are an excellent linguist. Please help me with the task of nested named entity recognition. Given an entity type and a sentence, please help me identify the positions of words in the sentence that are likely to be entities of this type. \nThe input sentence format is similar to a Python list, where each word or punctuation mark in the sentence is separately split and stored as a string in the list. The output format is the positions in the input list where you think the entity words are located.  \nThe current entity type that needs to be recognized is {entity_type}, which means {type_description}. \n Here are some examples. \n'''

    return prompt


def get_llm_revise_prompt():

    prompt = '''You are an excellent linguist. Please help me with the task of nested named entity recognition. I want to identify these entities in a sentence: [GPE, ORG, PER, LOC, WEA, VEH, FAC]. Here is detail explanation of there entity types:

    GPE: geographical political entities are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people;
    ORG: organization entities are limited to companies, corporations, agencies, institutions and other groups of people; 
    PER: a person entity is limited to human including a single individual or a group;
    LOC: location entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations; 
    WEA: weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder;
    VEH: vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles;
    FAC: facility entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges.\n
    '''

    return prompt


def get_llm_revise_prompt_we(examples, data_type='ace'):
    if data_type[:3]=='ace':
        gpe = str(examples['GPE'])
        org = str(examples['ORG'])
        per = str( examples['PER'])
        loc = str( examples['LOC'])
        wea = str( examples['WEA'])
        veh = str( examples['VEH'])
        fac = str( examples['FAC'])

        prompt = f'''You are an excellent linguist. Please help me with the task of nested named entity recognition. I want to identify these entities in a sentence: [GPE, ORG, PER, LOC, WEA, VEH, FAC]. Here is detail explanation of the entity types and corresponding examples:

        GPE: geographical political entities are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people. For example: {gpe};
        ORG: organization entities are limited to companies, corporations, agencies, institutions and other groups of people. For example: {org}; 
        PER: a person entity is limited to human including a single individual or a group. For example: {per};
        LOC: location entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations. For example: {loc}; 
        WEA: weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder. For example: {wea};
        VEH: vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles. For example: {veh};
        FAC: facility entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges. For example: {fac}\n\n'''
    elif data_type=='genia':
        keys = list(examples.keys())
        total_type = '['
        for i in keys:
            total_type += str(i)
            total_type += ', '
        total_type = total_type[:-2]
        total_type += ']'

        prompt = f'''You are an excellent linguist. Please help me with the task of nested named entity recognition. I want to identify these entities in a sentence: {total_type}. Here is examples entity types: \n'''

        for key, value in examples.items():
            temp = str(key) + ': ' + str(value)
            prompt += temp + '\n'

        prompt += '\n'
        
    else:
        print("[ERROR] data_type should be ace or genia!")
        assert(0)



    return prompt







class Prompt():
    def __init__(self,):
        return

    def entity_type_discript(self, data_type):

        if data_type == 'ace04' or data_type == 'ace05':
            result = { 'GPE':'geographical political entities are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people', 
            'ORG':' organization entities are limited to companies, corporations, agencies, institutions and other groups of people', 
            'PER':' a person entity is limited to human including a single individual or a group', 
            'LOC':' location entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations', 
            'WEA':'weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder', 
            'VEH':'vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles', 
            'FAC':' facility entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges'
            }


        return result

    def llm_revise_prompt_ace(self,):
        prompt = '''You are an excellent linguist. Please help me with the task of nested named entity recognition. I want to identify these entities in a sentence: [GPE, ORG, PER, LOC, WEA, VEH, FAC]. Here is detail explanation of there entity types:

        GPE: geographical political entities are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people;
        ORG: organization entities are limited to companies, corporations, agencies, institutions and other groups of people; 
        PER: a person entity is limited to human including a single individual or a group;
        LOC: location entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations; 
        WEA: weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder;
        VEH: vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles;
        FAC: facility entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges


        '''
        return prompt 


    def get_llm_revise_prompt(self, data_type):
        
        if data_type == 'ace04' or data_type == 'ace05':
            prompt = self.llm_revise_prompt_ace()

        return prompt
 
    