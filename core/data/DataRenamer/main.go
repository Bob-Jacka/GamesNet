/*
	*DataRename*
Программа для переименования файлов в директории на соответствующее имя.

*Защита от повреждения файловой системы не гарантируется :).

Version - 1.0.1
*/

package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	data_file_name = "datafile"
	data_file_ext  = ".png"
	fs_separate    = string(os.PathSeparator)
)

/*
Точка входа в программу.
*/
func main() {
	var args_len = len(os.Args)
	switch args_len {
	case 1:
		fmt.Println("Usage: go run main.go <Path to directory where you want to rename files>.")
	case 2:
		var dir_to_rename = os.Args[1]
		var file_names = ls(dir_to_rename)
		rename_one_by_one(file_names, dir_to_rename)
	default:
		fmt.Println("A lot of arguments, enter one argument.")
		os.Exit(0)
	}
}

/*
Функция для переименования файлов в директории.
File_names - массив имен в директории.
Dir_name - имя директории для переименования.
*/
func rename_one_by_one(file_names []string, dir_name string) {
	var file_number = 0
	sort.Strings(file_names)
	for _, file_name_with_ext := range file_names {
		if !is_dir(file_name_with_ext) {
			if is_png_image(file_name_with_ext) {
				var path_to_dir, _ = filepath.Abs(dir_name)
				var oldName_full_path = path_to_dir + fs_separate + file_name_with_ext
				var new_name_with_ext = data_file_name + strconv.Itoa(file_number) + "_" + data_file_ext
				var newName_full_path = path_to_dir + fs_separate + new_name_with_ext

				err := os.Rename(oldName_full_path, newName_full_path)
				if err != nil {
					fmt.Println("Error in renaming file:", err)
				} else {
					fmt.Printf("File %s renamed into %s\n", oldName_full_path, newName_full_path)
				}
				file_number++
			}
		}
	}
}

/*
Проверка того, что переданный путь это директория.
File_name - имя файла для проверки.
*/
func is_dir(file_name string) bool {
	fi, err := os.Stat(file_name)
	if err != nil {
		fmt.Println(err)
		return false
	}
	switch mode := fi.Mode(); {
	case mode.IsDir():
		fmt.Println("It is not a file, it is Directory.")
		return true
	case mode.IsRegular():
		fmt.Println("File")
		return false
	}
	return false
}

/*
Получение всех имен файлов в директории.
Which_dir - имя директории для сбора данных.
*/
func ls(which_dir string) []string {
	var file_slice []string
	entries, err := os.ReadDir(which_dir)
	if err != nil {
		log.Fatal(err)
	}
	for _, e := range entries {
		file_slice = append(file_slice, e.Name())
	}
	return file_slice
}

/*
Функция для проверки, что переданное имя это картинка
*/
func is_png_image(file_name string) bool {
	return strings.HasSuffix(file_name, data_file_ext) && strings.Contains(file_name, data_file_ext)
}
