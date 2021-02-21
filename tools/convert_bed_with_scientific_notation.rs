/*
Change scientific notation numbers (1.2e6) to standard notation (1200000)
on a per row basis in a BedGraph file. Input file is in line 14 and output
is in line 11.
*/
use std::fs::File;
use std::io::{self, BufRead, LineWriter, Write};
use std::path::Path;

fn main() {
    let new_file = Path::new("/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Theoretical_EcoDam/New folder/Hg38.EcoDam.insilico.canonical.chromnames_sort_fix.bed");
    let mut new_file = LineWriter::new(File::create(new_file).unwrap());
    // File hosts must exist in current path before this produces output
    if let Ok(lines) = read_lines("/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Theoretical_EcoDam/New folder/Hg38.EcoDam.insilico.canonical.chromnames_sort.bed") {
        // Consumes the iterator, returns an (Optional) String
        for line in lines {
            if let Ok(ip) = line {
                let mut split: Vec<_> = ip.split('\t').collect();
                let start = split[1].parse::<f64>().unwrap().to_string();
                split[1] = &start;
                let end = split[2].parse::<f64>().unwrap().to_string();
                split[2] = &end;
                let new_str = [split.join("\t"), "\n".to_string()].concat();
                new_file.write_all(new_str.as_bytes()).unwrap();
            }
        }
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
