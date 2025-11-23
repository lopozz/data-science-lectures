UNANSWERABLE_Q_GENERATION_PROMPT = """Vorrei che tu agissi come un generatore di domande. Ti fornirò un testo di riferimento e dovrai generare una sola domanda di tipo fattuale che NON può essere risposta utilizzando il contenuto fornito.

Le regole da seguire sono le seguenti:
- La domanda deve essere concisa e chiara.
- La domanda deve essere pertinente al tema del testo di riferimento ma richiede informazioni che vanno oltre il contenuto fornito
- La risposta richiede fatti, dati o informazioni assenti e non presenti nel testo.

---

Testo di riferimento:
Nel 1961, Yuri Gagarin divenne il primo uomo a viaggiare nello spazio a bordo della navicella Vostok 1. La missione durò 108 minuti e completò un’orbita attorno alla Terra. Il successo sovietico rappresentò un passo fondamentale nella corsa allo spazio e un trionfo propagandistico durante la Guerra Fredda.

Domanda:
Quali furono le reazioni ufficiali della NASA immediatamente dopo l’annuncio del volo di Gagarin?

---

Testo di riferimento:
Nel 1990, Tim Berners-Lee sviluppò il primo prototipo del World Wide Web presso il CERN di Ginevra. Il sistema permetteva di collegare documenti ipertestuali tramite link, aprendo la strada alla comunicazione globale e alla nascita di internet come la conosciamo oggi.

Domanda:
Quanti utenti attivi utilizzavano il Web nel suo primo mese di sperimentazione interna al CERN?

---

Testo di riferimento:
Durante il Rinascimento italiano, artisti come Leonardo da Vinci e Michelangelo contribuirono alla diffusione di un nuovo ideale di bellezza basato sull’armonia, la proporzione e l’osservazione della natura. Le loro opere influenzarono profondamente l’arte europea per secoli.

Domanda:
Quali tecniche pittoriche di Leonardo furono riprese dai pittori fiamminghi nel secolo successivo?

---

Testo di riferimento:
{ref_document}

Domanda:
"""


UNANSWERABLE_A_GENERATION_PROMPT = """Ti fornirò una domanda e un testo di riferimento.

La domanda NON può essere risposta utilizzando solo il testo fornito.

Genera una risposta che segua queste linee guida:
- Riconoscere chiaramente che le informazioni necessarie non sono presenti nel contesto.
- Spiega perchè non è possibile rispondere alla domanda facendo riferimento al contenuto del contesto.

Testo di riferimento:
{ref_document}

Domanda:
{question}

Risposta:
"""
