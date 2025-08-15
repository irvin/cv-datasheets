import sys, re, json

metadata_file = sys.argv[1] 
template_file = sys.argv[2]
output_dir = sys.argv[3]

metadata = json.loads(open(metadata_file).read())
template = open(template_file).read()
for locale in metadata:
	print(locale, metadata[locale])
	english_name = metadata[locale]['english_name']
	native_name = metadata[locale]['native_name']
	if native_name == '':
		native_name = '<' + english_name + '>'
	filled_template = template.replace('{{LOCALE}}', locale)
	filled_template = filled_template.replace('{{ENGLISH_NAME}}', english_name)
	filled_template = filled_template.replace('{{NATIVE_NAME}}', native_name)

	output_fd = open(output_dir + '/' + locale + '.md', 'w+')
	print(filled_template, file=output_fd)
	output_fd.close()
