from mtranslate import translate

LANGUAGES = ['af', 'pt']
TO_TRANSLATE = "I don't know we'll see"

def back_and_forth(txt, dest_lan):
    translated_txt = translate(txt, dest_lan)
    return translate(translated_txt)

def main(txt=TO_TRANSLATE, languages=LANGUAGES):
    for l in languages:
        print(back_and_forth(txt, l))

if __name__ == '__main__':
    main()