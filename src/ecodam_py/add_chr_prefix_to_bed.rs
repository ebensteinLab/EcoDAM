use std::fs::File;
use std::io::{self, BufRead, LineWriter, Write};
use std::path::Path;

fn main() {
    let new_file = Path::new("/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Theoretical_EcoDam/New folder/Hg38.EcoDam.insilico.canonical.keynames.bed");
    let mut new_file = LineWriter::new(File::create(new_file).unwrap());
    // File hosts must exist in current path before this produces output
    if let Ok(mut lines) = read_lines("/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Theoretical_EcoDam/New folder/Hg38.EcoDam.insilico.canonical.keynames.txt.txt") {
        // Consumes the iterator, returns an (Optional) String
        let _ = lines.next();
        for line in lines {
            if let Ok(ip) = line {
                let mut split: Vec<_> = ip.split('\t').collect();
                let chraddition = ["chr".to_string(), split[0].to_string()].concat();
                split[0] = &chraddition;
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

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
