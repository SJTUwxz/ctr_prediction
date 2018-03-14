#coding=utf-8
# use iconv to encode related txt files first

import logging
import os.path
import sys
import re
import jieba
# from importlib import reload

# reload(sys)

def reTest(content):
   # remove tags 
  reContent = re.sub('<contenttitle>|</contenttitle>','',content)
  reContent = re.sub('<doc>|</doc>', '', reContent)
  reContent = re.sub('<url>|</url>', '', reContent)
  reContent = re.sub('<docno>|</docno>', '', reContent)
  reContent = re.sub('<content>|</content>', '', reContent)
   # remove non-Chinese words
  reContent = reContent.decode( 'utf-8')
  filtrate = re.compile(u'[^\u4E00-\u9FA5]') 
  reContent = filtrate.sub(r'', reContent)
  reContent = reContent.encode('utf-8') 
  # remove digits
  reContent = re.sub("\d","",reContent) 
  # remove space
  reContent = re.sub("\s","",reContent) 

  return reContent

if __name__ == '__main__':
  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)
  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logger.info("running %s" % ' '.join(sys.argv))

   # check and process input arguments
  if len(sys.argv) < 3:
    print(globals()['__doc__'] % locals())
    sys.exit(1)
  inp, outp = sys.argv[1:3]
  space = " "
  i = 0
  f = open('./data/stopwords.txt.utf8','r')
  stopwords = [l.strip() for l in f.readlines()]

  finput = open(inp )
  foutput = open(outp,'w')
  for line in finput:
    line_seg = jieba.cut(reTest(line))
    new_line = []
    for word in line_seg:
        word = word.encode( 'utf-8') 
        if word not in stopwords:
            new_line.append(word) 
        else:
            print word
    foutput.write(space.join(new_line)) 
    i = i + 1
    if (i % 1000 == 0):
      logger.info("Saved " + str(i) + " articles_seg")
    foutput.write('\n')

  finput.close()
  foutput.close()
  logger.info("Finished Saved " + str(i) + " articles")



