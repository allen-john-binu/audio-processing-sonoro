from vicon_dssdk import ViconDataStream

VICON_SERVER = "localhost:801"

client = ViconDataStream.Client()

print("Connecting...")
client.Connect(VICON_SERVER)
print("Connected!")

# Enable the streams
client.EnableSegmentData()
client.EnableMarkerData()
client.EnableUnlabeledMarkerData()

print("\nWaiting for frame...")
client.GetFrame()

print("Frame:", client.GetFrameNumber())

print("\nSubjects available through DataStream:")

try:
    subjects = client.GetSubjectNames()

    print("Number of subjects:", len(subjects))

    for subject in subjects:
        print("  Subject:", subject)

        try:
            segments = client.GetSegmentNames(subject)

            for segment in segments:
                print("    Segment:", segment)

                try:
                    result = client.GetSegmentGlobalTranslation(
                        subject,
                        segment
                    )

                    print("      Translation:", result[0])
                    print("      Occluded:", result[1])

                except Exception as e:
                    print("      Translation ERROR:", repr(e))

        except Exception as e:
            print("    Segment ERROR:", repr(e))

except Exception as e:
    print("Subject ERROR:", repr(e))