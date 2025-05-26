

LLM_PROMPTS = {
    "numclaim": {
        "zeroshot": "Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. 'INCLAIM' refers to predictions or expectations about financial outcomes, it can be thought of as 'financial forecasts'. 'OUTOFCLAIM' refers to sentences that provide numerical information or established facts about past financial events. Now, for the following sentence provide the label in the first line and provide a short explanation in the second line. The sentence: ",
        "fewshot": "Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. 'INCLAIM' refers to predictions or expectations about financial outcomes, it can be thought of as 'financial forecasts'. 'OUTOFCLAIM' refers to sentences that provide numerical information or established facts about past financial events. " + "Here are two examples: \
                                \nExample 1: consolidated total capital was $2.9 billion for the quarter. // OUTOFCLAIM\
                                \nExample 2: we expect revenue growth to be in the range of 5.5% to 6.5% year on year. // INCLAIM " + "\nNow, for the following sentence provide the label in the first line and provide a short explanation in the second line. The sentence: "
    },

    "trec": {
        "zeroshot": "For the following question, which belongs to a specific category, categorize it into one of the following classes based on the type of answer it requires: Abbreviation (ABBR), Entity (ENTY), Description (DESC), Human (HUM), Location (LOC), Numeric (NUM). Provide the label in the first line and provide a short explanation in the second line. The question: ",
        "fewshot": "For the following question, which belongs to a specific category, categorize it into one of the following classes based on the type of answer it requires: Abbreviation (ABBR), Entity (ENTY), Description (DESC), Human (HUM), Location (LOC), Numeric (NUM). " + "Here are six examples:" + \
        "\nExample 1: how did serfdom develop in and then leave russia ? // DESC\
         \nExample 2: what films featured the character popeye doyle ? // ENTY\
         \nExample 3: what contemptible scoundrel stole the cork from my lunch ? // HUM\
         \nExample 4: what is the full form of .com ? // ABBR\
         \nExample 5: what sprawling u.s. state boasts the most airports ? // LOC\
         \nExample 6: when was ozzy osbourne born ? // NUM " + \
        "\nNow for the following question provide the label in the first line and provide a short explanation in the second line. The question: "
    },

    "semeval": {
        "zeroshot": "The task is to identify the type of semantic relationship between two nominals in a given sentence. Below are the definitions of the nine relationship categories you must choose from:\n" + \
                    "Cause-Effect (CE): An event or object leads to an effect.\n" + \
                    "Instrument-Agency (IA): An agent uses an instrument.\n" + \
                    "Product-Producer (PP): A producer causes a product to exist.\n" + \
                    "Content-Container (CC): An object is physically stored in a delineated area of space.\n" + \
                    "Entity-Origin (EO): An entity is coming or is derived from an origin (e.g., position or material).\n" + \
                    "Entity-Destination (ED): An entity is moving towards a destination.\n" + \
                    "Component-Whole (CW): An object is a component of a larger whole.\n" + \
                    "Member-Collection (MC): A member forms a nonfunctional part of a collection.\n" + \
                    "Message-Topic (MT): A message, written or spoken, is about a topic.\n" + \
                    "For the provided sentence below, determine the most accurate relationship category based on the descriptions provided. Respond by selecting the label (e.g., CE, IA, PP, etc.) that best matches the relationship expressed in the sentence. Provide the label in the first line and provide a short explanation in the second line." + \
                    "The sentence: ",
        "fewshot": "The task is to identify the type of semantic relationship between two nominals in a given sentence. Below are the definitions of the nine relationship categories you must choose from:\n" + \
                    "Cause-Effect (CE): An event or object leads to an effect. (Example: As the right front wheel of Senna 's car hit the wall , the violent impact caused a torsion on the steering column , causing it to break .)\n" + \
                    "Instrument-Agency (IA): An agent uses an instrument. (Example: The necromancer wields the power of death itself , a power no enemy can stand against .)\n" + \
                    "Product-Producer (PP): A producer causes a product to exist. (Example: This website , www.fertilityuk.org , shows how to interpret the changes that take place in the mucus secretions produced by the cells lining the cervix .)\n" + \
                    "Content-Container (CC): An object is physically stored in a delineated area of space. (Example: I sent you a suitcase with cash in it so you can fill it up with wine gummies .)\n" + \
                    "Entity-Origin (EO): An entity is coming or is derived from an origin (e.g., position or material) (Example: I have always felt so relieved that Roy and the boys had left the creek .).\n" + \
                    "Entity-Destination (ED): An entity is moving towards a destination. (Example: The machine blows water into the connecting conduit .)\n" + \
                    "Component-Whole (CW): An object is a component of a larger whole. (Example: He noticed a speck of blood on the man 's thumb and what he thought were several corresponding drops on the driver 's door of the truck .)\n" + \
                    "Member-Collection (MC): A member forms a nonfunctional part of a collection. (Example: With the conquest of Jerusalem in 1099 , Geoffrey de Bouillon established a chapter of secular canons in the basilica of the Holy Sepulcher to offer the sacred liturgy according to the Latin rite .)\n" + \
                    "Message-Topic (MT): A message, written or spoken, is about a topic. (Example: A number of scientific criticisms of Duesberg 's hypothesis were summarised in a review article in the journal Science in 1994 .)\n" + \
                    "For the provided sentence below, determine the most accurate relationship category based on the descriptions provided. Respond by selecting the label (e.g., CE, IA, PP, etc.) that best matches the relationship expressed in the sentence. Provide the label in the first line and provide a short explanation in the second line." + \
                    "The sentence: "
    },

    "20news": {
        "zeroshot": "The task is to classify the given text into one of the 20 news group categories. Below are the 20 categories you must choose from:\n" + \
                    "1. 'alt.atheism': Discussions related to atheism.\n" + \
                    "2. 'comp.graphics': Topics about computer graphics, including software and hardware.\n" + \
                    "3. 'comp.os.ms-windows.misc': Discussions about the Microsoft Windows operating system.\n" + \
                    "4. 'comp.sys.ibm.pc.hardware': Topics related to IBM PC hardware.\n" + \
                    "5. 'comp.sys.mac.hardware': Discussions about Mac hardware.\n" + \
                    "6. 'comp.windows.x': Topics about the X Window System.\n" + \
                    "7. 'misc.forsale': Posts related to buying and selling items.\n" + \
                    "8. 'rec.autos': Discussions about automobiles.\n" + \
                    "9. 'rec.motorcycles': Topics related to motorcycles.\n" + \
                    "10. 'rec.sport.baseball': Discussions about baseball.\n" + \
                    "11. 'rec.sport.hockey': Discussions about hockey.\n" + \
                    "12. 'sci.crypt': Topics about cryptography and encryption.\n" + \
                    "13. 'sci.electronics': Discussions about electronic systems and devices.\n" + \
                    "14. 'sci.med': Topics related to medical science and healthcare.\n" + \
                    "15. 'sci.space': Discussions about space and astronomy.\n" + \
                    "16. 'soc.religion.christian': Topics about Christianity and related discussions.\n" + \
                    "17. 'talk.politics.guns': Discussions about gun politics and related debates.\n" + \
                    "18. 'talk.politics.mideast': Topics about politics in the Middle East.\n" + \
                    "19. 'talk.politics.misc': General political discussions not covered by other categories.\n" + \
                    "20. 'talk.religion.misc': Discussions about miscellaneous religious topics.\n" + \
                    "For the provided text below, determine the most appropriate category based on the descriptions above. Respond by selecting the label (e.g., alt.atheism, comp.graphics, etc.) that best matches the topic of the text. Provide the label in the first line and a brief explanation in the second line." + \
                    "The text: ",
    }
}


LABEL_MAP = {
    "numclaim": {
        "outofclaim": 0,
        "inclaim": 1
    },

    "trec": {
        "desc": 0,
        "enty": 1,
        "hum": 2,
        "abbr": 3,
        "loc": 4,
        "num": 5
    },

    "semeval": {
        "ce": 0, 
        "cw": 1, 
        "cc": 2, 
        "ed": 3, 
        "eo": 4, 
        "ia": 5, 
        "mc": 6, 
        "mt": 7, 
        "pp": 8
    },

    "20news": {
        "alt.atheism": 0,
        "comp.graphics": 1,
        "comp.os.ms-windows.misc": 2,
        "comp.sys.ibm.pc.hardware": 3,
        "comp.sys.mac.hardware": 4, 
        "comp.windows.x": 5,
        "misc.forsale": 6, 
        "rec.autos": 7, 
        "rec.motorcycles": 8, 
        "rec.sport.baseball": 9, 
        "rec.sport.hockey": 10,
        "sci.crypt": 11,
        "sci.electronics": 12, 
        "sci.med": 13,
        "sci.space": 14, 
        "soc.religion.christian": 15,
        "talk.politics.guns": 16, 
        "talk.politics.mideast": 17, 
        "talk.politics.misc": 18, 
        "talk.religion.misc": 19
    }
}