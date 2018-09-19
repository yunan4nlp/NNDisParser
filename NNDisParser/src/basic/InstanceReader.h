#ifndef BASIC_INSTANCE_READER_H
#define BASIC_INSTANCE_READER_H

#include "Reader.h"
#include "N3LDG.h"
#include <sstream>

class InstanceReader : public Reader {
public:
  InstanceReader() {
  }

  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();
    static string strLine;
    static vector<string> vecLine;
    vecLine.clear();
    while (1) {
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (strLine.empty())
        break;
      vecLine.push_back(strLine);
    }

		const int lineSize = vecLine.size();
		int size = lineSize / 2;
		if (size == 0)
			return NULL;
		int info_size;
		vector<string> info;
		vector<string> word_pos;
		int start = 0, end = 0;
		string normal_word;
		for (int i = 0; i < size; i++) {
			split_bychar(vecLine[i], info, ' ');
			info_size = info.size() - 1;
			end = start + info_size; // include type.
			for (int j = 0; j < info_size; j++) {
				split_bychar(info[j], word_pos, '_');
				assert(word_pos.size() == 2);
				normal_word = normalize_to_lowerwithdigit(word_pos[0]);
				m_instance.words.push_back(normal_word);
				m_instance.tags.push_back(word_pos[1]);

				m_instance.total_text.push_back(word_pos[0]);
				m_instance.total_tags.push_back(word_pos[1]);
			}
			m_instance.total_text.push_back(info[info.size() - 1]);
			assert(info[info.size() - 1] == "<P>" || info[info.size() - 1] == "<S>");
			m_instance.total_tags.push_back(nullkey);
			auto sent_type = make_pair(make_pair(start, end), info[info.size() - 1]);
			m_instance.sent_types.push_back(sent_type);
			start = end + 1;
		}
		m_instance.syn_feats.resize(m_instance.words.size());
		assert(lineSize % 2 == 1 && lineSize - 1 > 0);
		m_instance.parse_tree(vecLine[lineSize - 1]);

    return &m_instance;
  }
};

#endif

