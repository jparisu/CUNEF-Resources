import streamlit as st
import re
import graphviz

# ==========================================
# 1. DATABASE & CONFIGURATION
# ==========================================

# Configuration for UI languages
UI_TEXT = {
    "English": {
        "title": "Esperanto Study Companion",
        "tab1": "Word Analyzer",
        "tab2": "Translator",
        "tab3": "Grammar Analyzer",
        "analyze_btn": "Analyze",
        "trans_btn": "Translate",
        "input_label": "Enter a word:",
        "input_sent_label": "Enter a sentence:",
        "root": "Root",
        "prefix": "Prefix",
        "suffix": "Suffix",
        "ending": "Ending",
        "desc": "Description",
        "trans": "Translation",
        "grammar_role": "Grammar Role",
        "tree_view": "Sentence Structure Tree",
        "examples": "Examples",
        "structure": "Structure Explanation",
        "parts": "Word Parts"
    },
    "Spanish": {
        "title": "Compañero de Estudio de Esperanto",
        "tab1": "Analizador de Palabras",
        "tab2": "Traductor",
        "tab3": "Analizador Gramatical",
        "analyze_btn": "Analizar",
        "trans_btn": "Traducir",
        "input_label": "Ingresa una palabra:",
        "input_sent_label": "Ingresa una frase:",
        "root": "Raíz",
        "prefix": "Prefijo",
        "suffix": "Sufijo",
        "ending": "Terminación",
        "desc": "Descripción",
        "trans": "Traducción",
        "grammar_role": "Función Gramatical",
        "tree_view": "Árbol de Estructura",
        "examples": "Ejemplos",
        "structure": "Explicación Estructural",
        "parts": "Partes de la palabra"
    }
}

# --- Esperanto Logic Data ---
# Standard Esperanto Prefixes
PREFIXES = {
    "bo": {"en": "relation by marriage (in-law)", "es": "parentesco por matrimonio (político)"},
    "dis": {"en": "separation/scattering", "es": "separación/dispersión"},
    "ek": {"en": "sudden/beginning action", "es": "acción repentina/comienzo"},
    "eks": {"en": "ex- (former)", "es": "ex- (anterior)"},
    "fi": {"en": "shameful/nasty", "es": "vergonzoso/desagradable"},
    "ge": {"en": "both sexes together", "es": "ambos sexos juntos"},
    "mal": {"en": "opposite", "es": "lo opuesto"},
    "mis": {"en": "incorrectly", "es": "incorrectamente"},
    "ne": {"en": "not/non-", "es": "no/in-"},
    "pra": {"en": "primordial/great- (relative)", "es": "primordial/bis- (pariente)"},
    "re": {"en": "again/back", "es": "otra vez/de vuelta"},
}

# Standard Esperanto Suffixes
SUFFIXES = {
    "aĉ": {"en": "pejorative (bad quality)", "es": "despectivo (mala calidad)"},
    "ad": {"en": "frequent/continuous action", "es": "acción frecuente/continua"},
    "aĵ": {"en": "concrete thing", "es": "cosa concreta"},
    "an": {"en": "member/inhabitant", "es": "miembro/habitante"},
    "ar": {"en": "collection/group", "es": "colección/grupo"},
    "ebl": {"en": "possible to...", "es": "posible de..."},
    "ec": {"en": "quality/abstract idea", "es": "calidad/idea abstracta"},
    "eg": {"en": "augmentative (big/intense)", "es": "aumentativo (grande/intenso)"},
    "ej": {"en": "place for...", "es": "lugar para..."},
    "em": {"en": "tendency/inclination", "es": "tendencia/inclinación"},
    "et": {"en": "diminutive (small/slight)", "es": "diminutivo (pequeño)"},
    "id": {"en": "offspring", "es": "descendencia"},
    "ig": {"en": "to make (causative)", "es": "hacer (causativo)"},
    "iĝ": {"en": "to become", "es": "hacerse/volverse"},
    "il": {"en": "tool/instrument", "es": "herramienta/instrumento"},
    "in": {"en": "female", "es": "femenino"},
    "ind": {"en": "worthy of", "es": "digno de"},
    "ism": {"en": "doctrine/system", "es": "doctrina/sistema"},
    "ist": {"en": "professional/enthusiast", "es": "profesional/aficionado"},
    "obl": {"en": "multiple", "es": "múltiplo"},
    "on": {"en": "fraction", "es": "fracción"},
    "uj": {"en": "container/country", "es": "contenedor/país"},
    "ul": {"en": "person characterized by root", "es": "persona caracterizada por la raíz"},
}

# Grammatical Endings
ENDINGS = {
    "o": {"en": "Noun", "es": "Sustantivo", "pos": "NOUN"},
    "a": {"en": "Adjective", "es": "Adjetivo", "pos": "ADJ"},
    "e": {"en": "Adverb", "es": "Adverbio", "pos": "ADV"},
    "i": {"en": "Verb (Infinitive)", "es": "Verbo (Infinitivo)", "pos": "VERB"},
    "as": {"en": "Verb (Present)", "es": "Verbo (Presente)", "pos": "VERB"},
    "is": {"en": "Verb (Past)", "es": "Verbo (Pasado)", "pos": "VERB"},
    "os": {"en": "Verb (Future)", "es": "Verbo (Futuro)", "pos": "VERB"},
    "us": {"en": "Verb (Conditional)", "es": "Verbo (Condicional)", "pos": "VERB"},
    "u": {"en": "Verb (Imperative)", "es": "Verbo (Imperativo)", "pos": "VERB"},
    "j": {"en": "Plural", "es": "Plural", "pos": "PLURAL"},
    "n": {"en": "Accusative (Object)", "es": "Acusativo (Objeto)", "pos": "ACC"},
}

# Mini Dictionary (In a real app, load this from a JSON/CSV)
DICTIONARY = {
    "amik": {"en": "friend", "es": "amigo"},
    "bel": {"en": "beautiful", "es": "bello"},
    "grand": {"en": "big", "es": "grande"},
    "rapid": {"en": "fast", "es": "rápido"},
    "lern": {"en": "learn", "es": "aprender"},
    "parol": {"en": "speak", "es": "hablar"},
    "hund": {"en": "dog", "es": "perro"},
    "kat": {"en": "cat", "es": "gato"},
    "lib": {"en": "book", "es": "libro"},
    "kant": {"en": "sing", "es": "cantar"},
    "esper": {"en": "hope", "es": "esperar"},
    "facil": {"en": "easy", "es": "fácil"},
    "lingv": {"en": "language", "es": "idioma"},
    "hom": {"en": "human", "es": "humano"},
    "tag": {"en": "day", "es": "día"},
    "bon": {"en": "good", "es": "bueno"},
    "labor": {"en": "work", "es": "trabajar"},
    "san": {"en": "health", "es": "salud"},
    "am": {"en": "love", "es": "amar"},
}

EXAMPLES = {
    "amik": "Mia amiko estas bona. (My friend is good.)",
    "lern": "Mi lernas Esperanton. (I learn Esperanto.)",
    "san": "Malsanulejo estas por malsanuloj. (A hospital is for sick people.)",
}

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================

def get_word_details(word, lang_code):
    """
    Deconstructs an Esperanto word into prefixes, root, suffixes, and endings.
    """
    clean_word = word.lower().strip().replace(".", "").replace(",", "")
    parts = []

    # 1. Strip grammatical endings (Right to Left)
    grammar_ending = ""
    remainder = clean_word

    # Check for plural/accusative first (-j, -n, -jn)
    if remainder.endswith("jn"):
        parts.insert(0, {"text": "jn", "type": "ending", "desc": "Plural Accusative"})
        remainder = remainder[:-2]
    elif remainder.endswith("n"):
        parts.insert(0, {"text": "n", "type": "ending", "desc": ENDINGS["n"][lang_code]})
        remainder = remainder[:-1]
    elif remainder.endswith("j"):
        parts.insert(0, {"text": "j", "type": "ending", "desc": ENDINGS["j"][lang_code]})
        remainder = remainder[:-1]

    # Check for POS ending
    found_pos = False
    for end in ["as", "is", "os", "us", "u", "i", "o", "a", "e"]:
        if remainder.endswith(end):
            parts.insert(0, {"text": end, "type": "ending", "desc": ENDINGS[end][lang_code]})
            remainder = remainder[:-len(end)]
            found_pos = True
            break

    # 2. Strip Prefixes (Left to Right)
    while True:
        found_prefix = False
        for pre in PREFIXES:
            if remainder.startswith(pre):
                # Heuristic: verify remainder after prefix is long enough to be a root
                if len(remainder) - len(pre) >= 2:
                    parts.insert(0 if not parts else len(parts)-len([p for p in parts if p['type'] == 'ending']),
                                 {"text": pre, "type": "prefix", "desc": PREFIXES[pre][lang_code]})
                    remainder = remainder[len(pre):]
                    found_prefix = True
                    break
        if not found_prefix:
            break

    # 3. Strip Suffixes (Right to Left from remainder)
    while True:
        found_suffix = False
        for suf in SUFFIXES:
            if remainder.endswith(suf):
                if len(remainder) - len(suf) >= 2:
                    # Insert before endings but after root (conceptually)
                    # For now, we add to a temporary list to append later
                    parts.insert(len([p for p in parts if p['type'] != 'ending' and p['type'] != 'suffix']),
                                 {"text": suf, "type": "suffix", "desc": SUFFIXES[suf][lang_code]})
                    remainder = remainder[:-len(suf)]
                    found_suffix = True
                    break
        if not found_suffix:
            break

    # 4. What is left is the ROOT
    root_meaning = "Unknown root"
    if remainder in DICTIONARY:
        root_meaning = DICTIONARY[remainder][lang_code]

    # Re-order parts list: Prefixes -> Root -> Suffixes -> Endings
    # The insert logic above is a bit messy, let's reconstruct cleanly

    final_breakdown = []

    # Extract prefixes from the temp list
    current_prefixes = [p for p in parts if p['type'] == 'prefix']
    final_breakdown.extend(current_prefixes)

    # Add Root
    final_breakdown.append({"text": remainder, "type": "root", "desc": f"{UI_TEXT['English']['root'] if lang_code=='en' else UI_TEXT['Spanish']['root']}: {root_meaning}"})

    # Extract suffixes (reverse order of finding) and endings
    current_suffixes = [p for p in parts if p['type'] == 'suffix']
    # Suffixes were found right-to-left, so we need to reverse them to be left-to-right
    current_suffixes.reverse()
    final_breakdown.extend(current_suffixes)

    current_endings = [p for p in parts if p['type'] == 'ending']
    final_breakdown.extend(current_endings)

    return final_breakdown, remainder

def simple_grammar_parser(sentence, lang_code):
    """
    A rule-based parser for Esperanto sentences.
    Returns tokens with POS tags and syntactic roles.
    """
    words = sentence.replace(".", "").replace(",", "").split()
    analysis = []

    has_subject = False

    for i, word in enumerate(words):
        details, root = get_word_details(word, lang_code)

        # Determine POS from details
        pos = "UNKNOWN"
        end_tags = [d['text'] for d in details if d['type'] == 'ending']

        is_plural = 'j' in end_tags or 'jn' in end_tags
        is_acc = 'n' in end_tags or 'jn' in end_tags

        role = "Unknown"

        if any(x in end_tags for x in ['as', 'is', 'os', 'us', 'u']):
            pos = "VERB"
            role = "Verb (Action)"
        elif 'i' in end_tags:
            pos = "VERB_INF"
            role = "Verb (Infinitive)"
        elif 'o' in end_tags or 'oj' in end_tags or 'on' in end_tags or 'ojn' in end_tags:
            pos = "NOUN"
            if is_acc:
                role = "Direct Object (Receive action)"
            else:
                # Simple heuristic: first nominative noun is subject
                if not has_subject:
                    role = "Subject (Doer)"
                    has_subject = True
                else:
                    role = "Nominative Complement"
        elif 'a' in end_tags or 'aj' in end_tags or 'an' in end_tags or 'ajn' in end_tags:
            pos = "ADJ"
            role = "Adjective (Describes noun)"
        elif 'e' in end_tags:
            pos = "ADV"
            role = "Adverb (Describes verb)"
        elif word.lower() in ["la"]:
            pos = "DET"
            role = "Article"
        elif word.lower() in ["kaj", "aŭ"]:
            pos = "CONJ"
            role = "Conjunction"
        elif word.lower() in ["en", "sur", "kun", "per", "al", "de"]:
            pos = "PREP"
            role = "Preposition"
        elif word.lower() in ["mi", "vi", "li", "ŝi", "ĝi", "ni", "ili"]:
            pos = "PRON"
            if is_acc:
                 role = "Direct Object"
            else:
                 role = "Subject"
                 has_subject = True

        analysis.append({
            "word": word,
            "root": root,
            "pos": pos,
            "role": role,
            "details": details
        })

    return analysis

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Esperanto Master", page_icon="💚", layout="wide")

# Sidebar - Language Selection
lang_select = st.sidebar.selectbox("Interface Language / Idioma", ["English", "Spanish"])
lang_code = "en" if lang_select == "English" else "es"
texts = UI_TEXT[lang_select]

st.title(f"💚 {texts['title']}")

# Tabs
tab1, tab2, tab3 = st.tabs([texts['tab1'], texts['tab2'], texts['tab3']])

# --- TAB 1: WORD ANALYZER ---
with tab1:
    st.header(texts['tab1'])
    word_input = st.text_input(texts['input_label'], key="word_ana")

    if st.button(texts['analyze_btn'], key="btn_ana"):
        if word_input:
            breakdown, root = get_word_details(word_input, lang_code)

            # 1. Visualization of Word Parts
            st.subheader(texts['parts'])
            cols = st.columns(len(breakdown))
            for idx, part in enumerate(breakdown):
                with cols[idx]:
                    st.markdown(f"**{part['text']}**")
                    color = "#e6f2ff" # default blueish
                    if part['type'] == 'root': color = "#ffd9b3" # orange
                    elif part['type'] == 'ending': color = "#d9ffcc" # green

                    st.markdown(f"<div style='background-color:{color}; padding:5px; border-radius:5px; font-size:0.8em; text-align:center'>{part['type'].upper()}</div>", unsafe_allow_html=True)
                    st.caption(part['desc'])

            st.divider()

            # 2. Translation & Description
            st.subheader(texts['desc'])
            # Reconstruct meaning based on parts
            meaning_str = ""
            if root in DICTIONARY:
                base_meaning = DICTIONARY[root][lang_code]
                meaning_str = f"**Base meaning:** {base_meaning}"

                # Dynamic meaning construction (simplified)
                mods = [p['desc'] for p in breakdown if p['type'] in ['prefix', 'suffix']]
                if mods:
                    meaning_str += f"\n\n**Modifiers applied:** {', '.join(mods)}"
            else:
                meaning_str = "Root not found in local dictionary."

            st.info(meaning_str)

            # 3. Examples
            st.subheader(texts['examples'])
            if root in EXAMPLES:
                st.success(EXAMPLES[root])
            else:
                st.write("No specific examples in database for this root.")

# --- TAB 2: TRANSLATOR ---
with tab2:
    st.header(texts['tab2'])

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        direction = st.radio("Direction", ["Esperanto -> " + lang_select, lang_select + " -> Esperanto"])

    trans_input = st.text_input("Text / Texto:", key="trans_in")

    if trans_input:
        res = []
        words = trans_input.lower().split()

        if direction.startswith("Esperanto"):
            # Esp -> Local
            for w in words:
                _, root = get_word_details(w, lang_code)
                if root in DICTIONARY:
                    res.append(DICTIONARY[root][lang_code])
                else:
                    res.append(f"[{w}?]")
            st.markdown(f"### {texts['trans']}:")
            st.write(" ".join(res))

        else:
            # Local -> Esp (Reverse lookup in dictionary)
            # This is a simple linear search for the demo
            found_words = []
            for w in words:
                found = False
                for k, v in DICTIONARY.items():
                    if v[lang_code] == w:
                        found_words.append(k + "o") # Assume noun form for simplicity
                        found = True
                        break
                if not found:
                    found_words.append(f"[{w}?]")

            st.markdown(f"### {texts['trans']}:")
            st.write(" ".join(found_words))

# --- TAB 3: GRAMMAR ANALYZER ---
with tab3:
    st.header(texts['tab3'])
    sent_input = st.text_area(texts['input_sent_label'], "La bona amiko parolas rapide.", height=100)

    if st.button(texts['analyze_btn'], key="btn_gram"):
        analysis = simple_grammar_parser(sent_input, lang_code)

        # 1. Interactive Word Breakdown (Hover-like effect using expanders)
        st.subheader(texts['parts'])
        for item in analysis:
            with st.expander(f"📖 {item['word']} ({item['role']})"):
                st.write(f"**Root:** {item['root']}")
                st.write("**Morphology:**")
                for p in item['details']:
                    st.text(f"- {p['text']}: {p['desc']}")

        st.divider()

        # 2. Dependency Tree Visualization
        st.subheader(texts['tree_view'])
        graph = graphviz.Digraph()
        graph.attr(rankdir='TB')

        # Find the verb (Root of the sentence usually)
        verb_node = None
        for i, item in enumerate(analysis):
            label = f"{item['word']}\n({item['role']})"
            graph.node(str(i), label, shape='box' if 'Verb' in item['role'] else 'ellipse')
            if "Verb" in item['role']:
                verb_node = i

        # Draw edges based on simple heuristics
        if verb_node is not None:
            for i, item in enumerate(analysis):
                if i == verb_node: continue

                # Subject connects to Verb
                if "Subject" in item['role']:
                    graph.edge(str(verb_node), str(i), label="subj")
                # Object connects to Verb
                elif "Object" in item['role']:
                    graph.edge(str(verb_node), str(i), label="obj")
                # Adverb connects to Verb
                elif "Adverb" in item['role']:
                    graph.edge(str(verb_node), str(i), label="mod")
                # Adjective connects to nearest Noun
                elif "Adjective" in item['role']:
                    # Find nearest noun
                    nearest = None
                    min_dist = 999
                    for j, sub_item in enumerate(analysis):
                        if "Noun" in sub_item['pos'] or "Subject" in sub_item['role'] or "Object" in sub_item['role']:
                            if abs(i-j) < min_dist:
                                min_dist = abs(i-j)
                                nearest = j
                    if nearest is not None:
                        graph.edge(str(nearest), str(i), label="amod")
                    else:
                        graph.edge(str(verb_node), str(i), label="?")
                else:
                    # Fallback
                    graph.edge(str(verb_node), str(i))

        st.graphviz_chart(graph)

        # 3. Structural Explanation
        st.subheader(texts['structure'])
        structure_text = ""
        subj = [x['word'] for x in analysis if "Subject" in x['role']]
        verb = [x['word'] for x in analysis if "Verb" in x['role']]
        obj = [x['word'] for x in analysis if "Object" in x['role']]

        if lang_code == "en":
            structure_text += f"The sentence has a standard **SVO** structure (Subject-Verb-Object). "
            if subj: structure_text += f"The subject is **{subj[0]}** (who does the action). "
            if verb: structure_text += f"The action is **{verb[0]}**. "
            if obj: structure_text += f"The object receiving the action is **{obj[0]}** (marked by the -n ending). "
        else:
            structure_text += f"La frase tiene una estructura **SVO** (Sujeto-Verbo-Objeto). "
            if subj: structure_text += f"El sujeto es **{subj[0]}** (quien realiza la acción). "
            if verb: structure_text += f"La acción es **{verb[0]}**. "
            if obj: structure_text += f"El objeto que recibe la acción es **{obj[0]}** (marcado por la terminación -n). "

        st.write(structure_text)
