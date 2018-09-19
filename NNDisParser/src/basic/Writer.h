#ifndef BASIC_WRITER_H
#define BASIC_WRITER_H

#include <fstream>
#include <iostream>

class Writer {
public:
  Writer() {
  }

  virtual ~Writer() {
    if (m_outf.is_open()) m_outf.close();
  }

  inline int startWriting(const char *filename) {
    m_outf.open(filename);
    if (!m_outf) {
      cout << "Writerr::startWriting() open file err: " << filename << endl;
      return -1;
    }
    return 0;
  }

  inline void finishWriting() {
    m_outf.close();
  }

  virtual int write(const Instance *pInstance) = 0;

  virtual int write(const CResult &result) = 0;

protected:
  ofstream m_outf;
};

#endif

