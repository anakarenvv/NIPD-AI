from my_data_generator import FILES_and_LABELS, CustomDataGen

# Pequeño subset para probar rutas de RABIES
subjects = [1, 2, 49, 57]   # Algunos sujetos del batch-001 y batch-002
sessions = [1]              # Solo sesión 1 para prueba rápida

# Creamos la instancia para obtener los archivos de RABIES
fl = FILES_and_LABELS(subjects, sessions, MRI_type='func', functional_type='dist')
pairs = fl.get_mask_and_bold()   # Devuelve [[image_abs_path, mask_abs_path], ...]

print("Total pares (image, mask) encontrados:", len(pairs))
if len(pairs) > 0:
    print("Ejemplo image:", pairs[0][0])
    print("Ejemplo mask :", pairs[0][1])

# Ahora creamos un generador mínimo usando esos pares
gen = CustomDataGen(
    df=pairs,
    batch_size=1,            # 1 sesión por batch
    subbatch_size=30,        # sub-batches de 30 vols
    vols=30,                 # usa solo 30 volúmenes
    format="just_brain",     # usa (mask * bold) + crop
    classes="sessions",      # clasificación por sesión
    num_class=3,             # 3 clases posibles: ses-01, ses-02, ses-03
    shuffle=False,
    functional_type="dist"
)

# Cargamos el primer sub-batch
X, y = gen[0]

# Mostramos las formas de los datos cargados
print("X shape:", getattr(X, "shape", (len(X),) + tuple(getattr(X[0], "shape", []))))
print("y shape:", y.shape)
