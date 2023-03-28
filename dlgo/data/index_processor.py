# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import multiprocessing
import six
if sys.version_info[0] == 3:
    from urllib.request import urlopen, urlretrieve
else:
    from urllib import urlopen, urlretrieve


def worker(url_and_target):  # Распараллеливание загрузки данных с помощью многопроцессорной обработки
    try:
        (url, target_path) = url_and_target
        print('>>> Загрузка ' + target_path)
        urlretrieve(url, target_path)
    except (KeyboardInterrupt, SystemExit):
        print('>>> Выход из дочернего процесса')


class KGSIndex:

    def __init__(self,
                 kgs_url='http://u-go.net/gamerecords/',
                 index_page='kgs_index.html',
                 data_directory='data'):
        """Создайте индекс zip-файлов, содержащих SGF-данные реальных игр Go на KGS.

        Параметры:
        -----------
        kgs_url: URL-адрес со ссылками на zip-файлы игр
        index_page: Имя локального html-файла kgs_url
        data_directory: имя каталога относительно текущего пути для хранения данных SGF
        """
        self.kgs_url = kgs_url
        self.index_page = index_page
        self.data_directory = data_directory
        self.file_info = []
        self.urls = []
        self.load_index()  # Загрузка индекса при создании.

    def download_files(self):
        """Загружаем zip-файлы, распределяя работу на все доступные процессоры CPU"""
        if not os.path.isdir(self.data_directory):
            os.makedirs(self.data_directory)

        urls_to_download = []
        for file_info in self.file_info:
            url = file_info['url']
            file_name = file_info['filename']
            if not os.path.isfile(self.data_directory + '/' + file_name):
                urls_to_download.append((url, self.data_directory + '/' + file_name))
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        try:
            it = pool.imap(worker, urls_to_download)
            for _ in it:
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print(">>> Пойман KeyboardInterrupt, завершающий работу рабочих")
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def create_index_page(self):
        """Если нет локального html-файла, содержащего ссылки на файлы, создайте его."""
        if os.path.isfile(self.index_page):
            print('>>> Чтение кэшированной индексной страницы')
            index_file = open(self.index_page, 'r')
            index_contents = index_file.read()
            index_file.close()
        else:
            print('>>> Загрузка индексной страницы')
            fp = urlopen(self.kgs_url)
            data = six.text_type(fp.read())
            fp.close()
            index_contents = data
            index_file = open(self.index_page, 'w')
            index_file.write(index_contents)
            index_file.close()
        return index_contents

    def load_index(self):
        """Создание фактическое представление индекса из ранее загруженного или кэшированного html-кода."""
        index_contents = self.create_index_page()
        split_page = [item for item in index_contents.split('<a href="') if item.startswith("https://")]
        for item in split_page:
            download_url = item.split('">Download')[0]
            if download_url.endswith('.tar.gz'):
                self.urls.append(download_url)
        for url in self.urls:
            filename = os.path.basename(url)
            split_file_name = filename.split('-')
            num_games = int(split_file_name[len(split_file_name) - 2])
            print(filename + ' ' + str(num_games))
            self.file_info.append({'url': url, 'filename': filename, 'num_games': num_games})


if __name__ == '__main__':
    index = KGSIndex()
    index.download_files()
