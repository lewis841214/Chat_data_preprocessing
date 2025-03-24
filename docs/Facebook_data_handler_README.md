# Facebook Data Handler

A module for processing and transforming Facebook message data downloaded from "Download Your Information" feature into a standardized format for further processing.

## Overview

This handler processes the Facebook message JSON files downloaded from Facebook's "Download Your Information" feature. It extracts conversation data, identifies roles (User vs Assistant), and converts the data into a standardized format that can be used for further processing or training.

## Features

- Processes Facebook message data from the inbox directory structure
- Automatically identifies user and assistant roles in conversations
- Converts timestamps to ISO format
- Standardizes conversation data to match the required input format
- Preserves original filenames in the output files
- Uses configurable platform identifier in the output
- Saves formatted data to a specified output directory

## Prerequisites

- Python 3.6+
- PyYAML (for YAML config support): `pip install pyyaml`

## Configuration

Configuration can be provided in either YAML or JSON format. Here's an example of the configuration settings:

```yaml
# Facebook data processing configuration

# Handler type
type: facebook

# Platform identifier
platform: facebook

# Path to Facebook data directory
platform_data_path: data/this_profile's_activity_across_facebook/messages/inbox/

# Path to save formatted output files
input_formated_path: data/formated_data/

# User's Facebook name (optional, will try to identify automatically if not provided)
# user_name: "Your Name"

# Processing settings
batch_size: 100 # Process conversations in batches
```

### Key Configuration Options

- `type`: Specifies the handler to use (keep as "facebook")
- `platform`: Platform identifier used in the output data structure
- `platform_data_path`: Path to the downloaded Facebook message files
- `input_formated_path`: Directory where processed files will be saved
- `user_name`: (Optional) Your Facebook username to help identify your messages

## Usage

### Download Your Facebook Data

1. Go to your Facebook account settings
2. Click on "Your Facebook Information"
3. Select "Download Your Information"
4. Choose the following settings:
   - Date Range: All time
   - Format: JSON
   - Media Quality: Low (unless you need high-quality media)
   - Make sure "Messages" is selected
5. Click "Create File" and wait for Facebook to prepare your download
6. Download the ZIP file when ready and extract it to your project directory

### Process the Data

Run the processing script:

```bash
python process_facebook_data.py --config configs/facebook_config.yaml
```

Optional parameters:

- `--config`, `-c`: Path to the configuration file (default: 'configs/facebook_config.yaml')
- `--user-name`, `-u`: Your Facebook username (to identify your messages)
- `--platform`, `-p`: Platform identifier to use in the output (default: facebook)

### Output Format

The processed data will be saved as JSON files in the directory specified by `input_formated_path`. The output files will have the same filenames as the original message files for easy traceability.

Each output file follows this structure:

```json
{
  "platform": "facebook",
  "conversation_id": "unique_identifier",
  "participants": ["Participant1", "Participant2"],
  "created_at": "2023-01-01T12:00:00",
  "last_message_at": "2023-01-02T14:30:00",
  "conversation": [
    {
      "role": "User",
      "content": "Hello, how are you?",
      "do_train": false
    },
    {
      "role": "Assistant",
      "content": "I'm doing well, thank you! How about you?",
      "do_train": true
    }
  ]
}
```

## Customization

The handler can be customized by modifying the `FacebookHandler` class in `platform/facebook_handler.py`. Some potential customizations include:

- Changing how user identification works (currently uses heuristics)
- Adding support for other message types (photos, videos, etc.)
- Implementing content filtering
- Handling reactions and other Facebook-specific features
- Customizing file naming and output structure 