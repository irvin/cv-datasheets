import sys, re

template_file = sys.argv[1]
datasheet_description = sys.argv[2]

table = {}
current_key = ''
for line in open(datasheet_description):
	if re.match('^==[^ ]+==$', line): 
		if current_key != '':
			table[current_key] = table[current_key].strip()

		current_key = line.strip().replace('=','')
		table[current_key] = ''
		continue

	if current_key != '':
		table[current_key] += line

template = open(template_file).read()

found_keys = list([re.sub('[{}]', '', k) for k in re.findall('{{[^}]+}}', template)])

for k in found_keys:
	if k in table:
		template = template.replace('{{'+k+'}}', table[k])
	else:
		template = template.replace('{{'+k+'}}', '')


template = re.sub('\n\n\n+', '\n\n', template)
print(template)
