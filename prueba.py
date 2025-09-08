from my_data_generator import FILES_and_LABELS, CustomDataGen

# Lista de sujetos (ajusta según los que tengas)
subjects = list(range(1, 6))     # sub-001 a sub-005
sessions = [1]                   # solo ses-01 para probar

# Cargar archivos .nii desde F:/rawdata/
fl = FILES_and_LABELS(subjects, sessions, MRI_type='func', functional_type='rest')
files_rel, labels = fl.get_ID_filenames()

print("Número de ejemplos encontrados:", len(files_rel))
print("Primer archivo relativo:", files_rel[0])

# Crear el generador de datos
gen = CustomDataGen(df=files_rel,
                    batch_size=1,
                    subbatch_size=30,
                    format="vol",          # puedes probar también con "rgb" o "grayscale"
                    classes="sessions",
                    num_class=3,
                    vols=30,
                    functional_type="rest")

# Probar primer subbatch
X, y = gen[0]

print("Forma de X:", X.shape)
print("Forma de y:", y.shape)
